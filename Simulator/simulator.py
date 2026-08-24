import os
import shlex
import subprocess
import re
import sys
import yaml
import time
import datetime
import threading
from pathlib import Path
import uuid

import torch
import numpy as np

from PyTorchSimFrontend import extension_config

# Configure logger for Simulator module
logger = extension_config.setup_logger()
from tqdm import tqdm


class ProgressBar:
    def __init__(self, desc, silent_mode=False, update_interval=0.5):
        self.desc = desc
        self.silent_mode = silent_mode
        self.update_interval = update_interval
        self.pbar = None
        self.finished = False
        self.progress_thread = None

    def __enter__(self):
        if not self.silent_mode:
            self.pbar = tqdm(
                desc=self.desc,
                bar_format='{desc}: {elapsed}',
                leave=False,  # Don't leave the bar when done (it will disappear)
                ncols=80,
                disable=False,
                total=100,  # Use a total for smooth animation
            )
            # Update progress bar in a separate thread
            def update_progress():
                while not self.finished:
                    self.pbar.update(1)
                    time.sleep(self.update_interval)

            self.progress_thread = threading.Thread(target=update_progress, daemon=True)
            self.progress_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished = True
        if not self.silent_mode and self.pbar is not None:
            self.pbar.close()
        return False


TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.uint8,
    torch.bfloat16: np.float16,
    torch.float16: np.float16,
}

#: What gem5 says when it dies, as opposed to the libc backtrace that follows it.
_GEM5_SIGNAL = re.compile(
    r"^(?:.*\b(?:panic|fatal):.*|.*Assertion.*|Program aborted at tick.*)$", re.M)


def _gem5_failure(proc, log_path):
    """The message a failed gem5 run should raise, from the log it redirects to.

    `-r` sends stdout AND stderr to `--stdout-file`, so the pipes are empty and
    a return code on its own says nothing: gem5 panics and aborts.
    """
    how = (f"signal {-proc.returncode}" if proc.returncode < 0
           else f"exit {proc.returncode}")
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        text = (proc.stderr or "") + (proc.stdout or "")
    hits = [h.strip() for h in _GEM5_SIGNAL.findall(text)]
    if not hits:
        hits = [l.strip() for l in text.strip().splitlines()[-3:]]
    if not hits:
        hits = [f"gem5 left no diagnostic; see {log_path}"]
    return (f"Gem5 Simulation Failed ({how}); full log in {log_path}\n  "
            + "\n  ".join(h[:300] for h in hits[:3]))


class CycleSimulator():
    def __init__(self) -> None:
        pass

    def compile_and_simulate(self, target_binary, vectorlane_size, silent_mode=False):
        dir_path = os.path.join(os.path.dirname(target_binary), "m5out")
        gem5_script_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "gem5_script/script_systolic.py")
        # --vlen too: script_systolic.py defaults to 256, while tnpu.cycle builds the
        # sample with +zvl<VLEN>b from the same config. A sample measured at a register
        # width the binary was not built for is not this machine's cycle count.
        gem5_cmd = [extension_config.CONFIG_GEM5_PATH, "-r", "--stdout-file=sto.log", "-d", dir_path, gem5_script_path,
                    "-c", target_binary, "--vlane", str(vectorlane_size),
                    "--vlen", str(extension_config.vpu_vector_length_bits)]

        if not silent_mode:
            logger.debug(f"[Gem5] cmd> {' '.join(gem5_cmd)}")
            logger.info("[Gem5] Gem5 simulation started")

        proc = subprocess.run(gem5_cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(_gem5_failure(proc, os.path.join(dir_path, "sto.log")))

        with open(f"{dir_path}/stats.txt", "r") as stat_file:
            raw_list = stat_file.readlines()
            cycle_per_tick = [int(line.split()[1]) for line in raw_list if "system.clk_domain.clock" in line][0]
            cycle_list = [int(line.split()[1]) for line in raw_list if "system.cpu.numCycles" in line]
        cycle_list = cycle_list[:-1]
        return cycle_list

class TOGSimulator():
    TOGSIM_RESULT_PATH_KEY = "TOGSIM_RESULT_PATH"
    FINISH_STR = "Simulation finished"
    ALLOC_POOL = dict() # For eagermode buffer plan
    _TOGSIM_CONFIG_ENV_UNSET = object()
    def __init__(self, config_path=None, togsim_path=None) -> None:
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        self.base_dir = togsim_path
        self.config_path = config_path
        self.config_yaml = self.load_yaml(self.config_path)
        self.process = None
        self._next_kernel_id = 0  # Auto-incrementing kernel ID

        # Create FIFOs for command and event communication
        self.fifo_dir = os.path.join("/tmp", f"togsim_fifo_{os.getpid()}")
        os.makedirs(self.fifo_dir, exist_ok=True)
        self.trace_file_path = os.path.join(self.fifo_dir, "cmd_fifo")
        self.trace_log = "# command_type, kernel_id, device_index, stream_index, tog_path, attribute_path, timestamp\n"

        # Create FIFOs if they don't exist
        if os.path.exists(self.trace_file_path):
            os.remove(self.trace_file_path)
        os.mkfifo(self.trace_file_path)

        # Start TOGSim process
        self._start_process()

        # Open trace file FIFO once and keep it open (after process starts)
        self._trace_file_lock = threading.Lock()
        try:
            self._trace_file_handle = open(self.trace_file_path, 'w')
        except IOError as e:
            logger.error(f"[TOGSim] Failed to open trace file: {e}")
            raise RuntimeError(f"Failed to open trace file: {e}")

    def __enter__(self):
        """Context manager entry.

        Sets ``TOGSIM_CONFIG`` to this instance's config path so that compilation
        (``extension_config`` / codegen) uses the same YAML as TOGSim. Previous
        value is restored in ``__exit__``.
        """
        if "TOGSIM_CONFIG" in os.environ:
            self._old_togsim_config_env = os.environ["TOGSIM_CONFIG"]
        else:
            self._old_togsim_config_env = self._TOGSIM_CONFIG_ENV_UNSET
        os.environ["TOGSIM_CONFIG"] = os.path.abspath(self.config_path)

        self.old_tog_simulator = torch.npu.get_tog_simulator()
        torch.npu.set_tog_simulator(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        self.until()
        torch.npu.set_tog_simulator(self.old_tog_simulator)

        if self._old_togsim_config_env is self._TOGSIM_CONFIG_ENV_UNSET:
            os.environ.pop("TOGSIM_CONFIG", None)
        else:
            os.environ["TOGSIM_CONFIG"] = self._old_togsim_config_env

    def _start_process(self):
        cmd = f"{self.get_togsim_command(self.config_path, self.base_dir)} --models_list {self.trace_file_path}"
        if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
            cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

        logger.debug(f"[TOGSim] cmd> {cmd}")
        if self.process is None:
            self.process = subprocess.Popen(
                shlex.split(cmd),
                #stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            logger.warning("[TOGSim] Simulator is already running.")

    def _cleanup_fifos(self):
        """Clean up FIFO files"""
        try:
            if os.path.exists(self.trace_file_path):
                os.remove(self.trace_file_path)
            if os.path.exists(self.fifo_dir):
                os.rmdir(self.fifo_dir)
        except OSError as e:
            logger.warning(f"[TOGSim] Failed to clean up FIFOs: {e}")

    def _send_command(self, command_type, device_index, stream_index, tog_path="", attribute_path="", timestamp=0):
        """
        Internal method to send a command to TOGSim via FIFO.

        Args:
            command_type: Type of command ("LAUNCH_KERNEL" or "DEVICE_SYNC")
            device_index: Device index
            stream_index: Stream index
            tog_path: kernel-dir handle; TOGSim derives trace.so/trace_cycles.tsv from
                its directory (the ONNX file itself is only read on the STONNE sparse
                path) - empty for DEVICE_SYNC
            attribute_path: Path to attribute file - empty for DEVICE_SYNC
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this command
        """
        if self.process is None:
            raise RuntimeError("[TOGSim] Simulator process is not running")

        if self.process.poll() is not None:
            raise RuntimeError("[TOGSim] Simulator process has terminated")

        # Get and increment kernel ID
        kernel_id = self._next_kernel_id
        self._next_kernel_id += 1

        # Format command: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
        command = f"{command_type},{kernel_id},{device_index},{stream_index},{tog_path},{attribute_path},{timestamp}"

        with self._trace_file_lock:
            # Write command to TOGSim
            try:
                self._trace_file_handle.write(command + '\n')
                self._trace_file_handle.flush()
                self.trace_log += command + '\n'
                logger.debug(f"[TOGSim] Sent command: {command}")
            except IOError as e:
                logger.error(f"[TOGSim] Failed to write to trace file: {e}")
                raise RuntimeError(f"Failed to send command to TOGSim: {e}")
        return kernel_id

    def until(self):
        # Make sure that all kernels in the stream are finished
        torch.npu.synchronize()

        # Close trace file handle if open
        if self._trace_file_handle is not None:
            try:
                self._trace_file_handle.close()
            except:
                pass
            self._trace_file_handle = None

        if self.process:
            self.process.wait()

            # Read output streams
            stdout_output = ""
            stderr_output = ""
            if self.process.stdout:
                stdout_output = self.process.stdout.read()
            if self.process.stderr:
                stderr_output = self.process.stderr.read()

            # Print stderr immediately if there's any error output
            if stderr_output:
                sys.stderr.write(stderr_output)
                sys.stderr.flush()

            # Save stdout to result file
            if stdout_output:
                result_path = extension_config.CONFIG_TORCHSIM_LOG_PATH
                os.makedirs(result_path, exist_ok=True)
                file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".log"
                result_path = os.path.join(result_path, file_name)
                with open(result_path, "w") as f:
                    f.write(stdout_output)
                logger.info(f'[TOGSim] Simulation log is stored to "{result_path}"')
            self.process = None

        # Save trace_log with same name but .trace extension
        if self.trace_log:
            result_path = extension_config.CONFIG_TORCHSIM_LOG_PATH
            os.makedirs(result_path, exist_ok=True)
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".trace"
            trace_path = os.path.join(result_path, file_name)
            with open(trace_path, "w") as f:
                f.write(self.trace_log)
            logger.info(f'[TOGSim] Trace log is stored to "{trace_path}"')

        # Clean up FIFOs
        self._cleanup_fifos()

    def launch_kernel(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        """
        Launch a kernel via FIFO communication.

        Args:
            device_index: Device index
            stream_index: Stream index
            tog_path: kernel-dir handle; TOGSim derives trace.so from its directory
                (the ONNX file itself is only read on the STONNE sparse path)
            attribute_path: Path to attribute file
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The kernel ID assigned to this launch
        """
        return self._send_command("LAUNCH_KERNEL", device_index, stream_index, tog_path, attribute_path, timestamp)

    def device_synchronize(self, device_index):
        """
        Synchronize all streams on a device via FIFO communication.

        Args:
            device_index: Device index to synchronize
            timestamp: Timestamp in nanoseconds (default: 0)

        Returns:
            int: The command ID assigned to this synchronization
        """
        # For device_synchronize, stream_index is not meaningful, use 0
        return self._send_command("DEVICE_SYNC", device_index, 0, "", "", 0)

    @classmethod
    def sram_alloc(cls, buf_name, addr_range):
        cls.ALLOC_POOL[buf_name] = addr_range

    @classmethod
    def sram_dealloc(cls, buf_name, addr_range):
        if buf_name in cls.ALLOC_POOL:
            del cls.ALLOC_POOL[buf_name]

    @staticmethod
    def write_kernel_attribute_file(attribute_dir, inputs, alloc_pool=None):
        """
        Write kernel attribute YAML (address_info + sram_alloc) under attribute_dir.

        Does not require a TOGSimulator instance. alloc_pool defaults to class ALLOC_POOL.

        Args:
            attribute_dir: Directory to hold numbered attribute files (created if needed)
            inputs: Kernel input tensors (data_ptr used for address_info)
            alloc_pool: Optional dict like ALLOC_POOL; defaults to TOGSimulator.ALLOC_POOL

        Returns:
            Path to the written YAML file.
        """
        if alloc_pool is None:
            alloc_pool = TOGSimulator.ALLOC_POOL
        address_info = {}
        sram_buffer = {}
        yaml_content = {}

        os.makedirs(attribute_dir, exist_ok=True)
        index = str(len(os.listdir(attribute_dir)))
        attribute_file = os.path.join(attribute_dir, index)

        for idx, tensor in enumerate(inputs):
            address_info[f"arg{idx}"] = tensor.data_ptr()
        yaml_content["address_info"] = address_info

        for buf_name, range in alloc_pool.items():
            sram_buffer[buf_name] = range
        yaml_content["sram_alloc"] = sram_buffer

        with open(attribute_file, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno())
        return attribute_file

    def load_yaml(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"YAML file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    def get_core_freq(self):
        if "core_freq_mhz" in self.config_yaml:
            return self.config_yaml["core_freq_mhz"] * 1000 * 1000 # MHz
        else:
            raise KeyError("Key 'core_freq' not found in JSON.")

    @staticmethod
    def get_togsim_command(config_path, togsim_path=None):
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")
        bin = os.path.join(togsim_path, "build/bin/Simulator")
        config = os.path.join(togsim_path, config_path)
        cmd = f"{bin} --config {config}"
        return cmd

    @staticmethod
    def run_standalone(
        model_path,
        attribute_path="",
        autotune_mode=False,
        config_path=None,
        togsim_path=None,
        timeout_sec=None,
    ):
        """
        Run a single kernel simulation in standalone mode.
        This method starts a new TOGSim process, runs the kernel, and waits for completion.
        For streaming multiple kernels, use launch_kernel() instead.

        Args:
            model_path: kernel-dir handle; trace.so/trace_cycles.tsv are derived from
                its directory (the ONNX file itself is only read on the STONNE sparse path)
            attribute_path: Path to attribute file
            autotune_mode: If True, run in autotune mode (silent)
            config_path: Path to TOGSim config file (required)
            togsim_path: Path to TOGSim directory (optional, defaults to CONFIG_TORCHSIM_DIR/TOGSim)
            timeout_sec: If set, terminate the Simulator subprocess after this many seconds
                (autotune uses this to skip very slow tile candidates).

        Returns:
            Path to the simulation result log file
        """
        if config_path is None:
            config_path = extension_config.CONFIG_TOGSIM_CONFIG
        if togsim_path is None:
            togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")

        # Create result path with appropriate filename
        if autotune_mode:
            base_dir = Path(model_path).parent / "togsim_result"
        else:
            base_dir = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)

        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        result_path = base_dir / f"{file_name}.log"
        trace_file_path = base_dir / f"{file_name}.trace"

        # Create trace file in result directory
        kernel_id, device_index, stream_index, timestamp = 0, 0, 0, 0
        command = f"LAUNCH_KERNEL,{kernel_id},{device_index},{stream_index},{model_path},{attribute_path},{timestamp}\n"
        with open(trace_file_path, 'w') as trace_file:
            trace_file.write(command)
            trace_file.flush()
            os.fsync(trace_file.fileno())

        try:
            # Drive the simulation from the emitted trace.so (the C++ TOG path). The
            # ONNX --models_list path remains only for callers without a trace.so (the
            # STONNE sparse path); the normal compile always emits one.
            trace_so = os.path.join(os.path.dirname(str(model_path)), "trace.so")
            cycle_tsv = os.path.join(os.path.dirname(str(model_path)), "trace_cycles.tsv")
            base_cmd = TOGSimulator.get_togsim_command(config_path, togsim_path)
            if os.path.exists(trace_so):
                cmd = f"{base_cmd} --trace_so {trace_so} --cycle_table {cycle_tsv}"
            else:  # ONNX TOG path (STONNE sparse)
                cmd = f"{base_cmd} --models_list {trace_file_path}"
            if extension_config.CONFIG_TOGSIM_DEBUG_LEVEL:
                cmd += f" --log_level {extension_config.CONFIG_TOGSIM_DEBUG_LEVEL}"

            if not autotune_mode:
                logger.debug(f"[TOGSim] cmd> {cmd}")
                logger.info("[TOGSim] TOGSim simulation started")
            with ProgressBar("[TOGSim] Running simulation", silent_mode=autotune_mode):
                completed = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    check=True,
                    timeout=timeout_sec,
                )
                result = completed.stdout
        except subprocess.TimeoutExpired as e:
            logger.warning(
                "[TOGSim] Simulator subprocess exceeded timeout (%.1f s); terminating.",
                float(timeout_sec) if timeout_sec is not None else -1.0,
            )
            raise RuntimeError("TOGSim subprocess timeout") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"[TOGSim] Command failed with exit code {e.returncode}")
            logger.error(f"[TOGSim] Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
            assert 0

        # Prevent race condition
        with open(result_path, "w") as f:
            f.write(result.decode())
            f.flush()
            os.fsync(f.fileno())

        if not autotune_mode:
            import logging as _logging
            model_path_log = f' of "{model_path}" ' if logger.isEnabledFor(_logging.DEBUG) else " "
            logger.info(f'[TOGSim] Simulation log{model_path_log}is stored to "{result_path}"')
        return result_path

    @staticmethod
    def get_result_from_file(result_path):
        core_metrics = {}
        dram_channel_bw = {}
        avg_dram_bw = 0.0
        simulation_time = float("inf")
        total_cycle = float("inf")

        # Read and find total stat position
        with open(result_path, "r") as f:
            lines = f.readlines()

        simulation_finished_idx = -1
        simulation_finished = False
        for idx, line in enumerate(lines):
            if TOGSimulator.FINISH_STR in line:
                simulation_finished = True
                simulation_finished_idx = idx
                break

        if simulation_finished_idx == -1:
            logger.warning(f"[TOGSim] Warning: Unable to parse the output file ({result_path}). The file may be improperly formatted.")
            return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time

        total_stat_lines = lines[simulation_finished_idx:]

        for line in total_stat_lines:
            # Parse core metrics (MatMul active cycle, Vector active cycle, etc.)
            if 'Core' in line:
                if 'MatMul active cycle' in line:
                    matmul_cycle = re.search(r'MatMul active cycle (\d+)', line).group(1)
                    vector_cycle = re.search(r'Vector active cycle (\d+)', line).group(1)
                    core_metrics['MatMul_active_cycle'] = int(matmul_cycle)
                    core_metrics['Vector_active_cycle'] = int(vector_cycle)
                elif 'Systolic Array Utilization' in line:
                    systolic_util = re.search(r'Systolic Array Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    vector_util = re.search(r'Vector Unit Utilization\(%\) (\d+\.?\d*)', line).group(1)
                    total_cycle = re.search(r'Total cycle: (\d+)', line).group(1)
                    core_metrics['Systolic_Array_Utilization'] = float(systolic_util)
                    core_metrics['Vector_Unit_Utilization'] = float(vector_util)
                    core_metrics['Total_cycle'] = int(total_cycle)

            # Parse DRAM channel bandwidth utilization
            if 'DRAM CH' in line:
                channel = re.search(r'DRAM CH\[(\d+)\]', line).group(1)
                bw_util = re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1)
                dram_channel_bw[f'CH[{channel}]'] = float(bw_util)

            # Parse average DRAM bandwidth
            if 'DRAM: AVG BW Util' in line:
                avg_dram_bw = float(re.search(r'AVG BW Util (\d+\.?\d*)%', line).group(1))

            if 'Total execution cycles' in line:
                total_cycle = int(re.search(r'Total execution cycles: (\d+)', line).group(1))

            # Parse total simulation time
            if 'Wall-clock time for simulation' in line:
                simulation_time = float(re.search(r'Wall-clock time for simulation: (\d+\.?\d*) seconds', line).group(1))
        return core_metrics, dram_channel_bw, avg_dram_bw, simulation_time, total_cycle

if __name__ == "__main__":
    # Example paths (adjust these to your actual test files)
    test_tog_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/tile_graph.onnx"
    test_attribute_path = "/workspace/PyTorchSim/outputs/6vxl6mwzhfl/runtime_0001/attribute/0"

    # Test: Launch multiple kernels
    sim = TOGSimulator(config_path="/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml")
    with sim:
        try:
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")

        try:
            id2 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id1 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
            id3 = torch.npu.launch_kernel(tog_path=test_tog_path, attribute_path=test_attribute_path)
        except Exception as e:
            print(f"Error during kernel launch: {e}")
    print(sim.trace_log)