#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <yaml-cpp/yaml.h>

enum class CoreType { WS_MESH, STONNE };

enum class DramType { SIMPLE, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class L2CacheType { NOCACHE, DATACACHE };

struct SimulationConfig {
  /* Path to the top-level hardware YAML passed to the simulator (empty if not from a file). */
  std::string config_file_path;

  /* Core config */
  std::vector<CoreType> core_type;
  std::string stonne_config_path;
  uint32_t num_cores;
  uint32_t core_freq_mhz;
  uint32_t core_print_interval = 0;
  uint32_t num_systolic_array_per_core = 1;
  uint32_t num_stonne_per_core = 1;
  uint32_t num_stonne_port = 1;
  // Per-core VMEM/spad capacity (KB) for the trace-path DMA throttle (sec 10.4): a
  // load that would overflow the spad waits for a consumer to free a tile. 0 = unset
  // -> disabled. Legacy TileGraphParser insts have alloc id -1 and are never gated.
  uint32_t core_spad_size_kb = 0;
  // SA weight-buffer depth (sec 10.4): weight tiles a systolic array holds; a
  // preload stalls until a slot frees (its matmuls finished). 2 = weight
  // double-buffer (convention default, tunable). 0 = disabled.
  uint32_t sa_weight_buffer_depth = 2;
  // A load whose DRAM tile the core's previous dispatch already brought in is not
  // fetched again: the next grid step reuses the resident block, as a Pallas grid
  // does when a block index does not change between consecutive steps.
  bool dma_reuse_across_dispatch = false;
  // Dispatches (work-items) a core runs at once when each fits its share of the spad.
  // 2 is a double buffer; a Pallas-style pipeline also overlaps the previous step's
  // write-back, which takes 3.
  uint32_t max_concurrent_dispatch = 2;
  // A dispatch left with only its stores to drain stops counting against
  // max_concurrent_dispatch: the next one starts while the write-back finishes, as a
  // pipeline that issues its output copy and moves on.
  bool release_dispatch_at_store = false;
  // A new dispatch starts only once every running one has its loads in: one step's
  // loads in flight at a time, as a pipeline that prefetches exactly the next step.
  bool dispatch_after_loads = false;
  // DMA streams take loads and stores in the order they were issued; false serves every
  // queued load before any store.
  bool dma_issue_order = false;
  // DMA streams in flight per core, and the requests one stream may inject per cycle
  // (fractional; 0 = no cap). 1 stream with no cap is one DMA at a time at the full
  // injection width; a TPU DMA runs at a fraction of HBM and several share it.
  uint32_t dma_streams = 1;
  double dma_stream_req_per_cycle = 0;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_num_partitions = 1;
  uint32_t dram_channels_per_partitions = 0;
  uint32_t dram_freq_mhz;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  float dram_bandwidth_gbps_per_channel = 0.f;
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* L2 Cache config */
  L2CacheType l2d_type = L2CacheType::NOCACHE;
  std::string l2d_config_str;
  uint32_t l2d_hit_latency = 1;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq_mhz;
  uint32_t icnt_latency;
  uint32_t icnt_stats_print_period_cycles=0;

  /* Sheduler config */
  uint32_t num_partition=1;
  std::string scheduler_type;

  /* Core id, Partiton id mapping */
  std::map<uint32_t, uint32_t> partiton_map;

  /* Other configs */
  std::string layout;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_dram_bandwidth() const {
    if (dram_bandwidth_gbps_per_channel > 0.f)
      return dram_bandwidth_gbps_per_channel * static_cast<float>(dram_channels);
    return 0.f;
  }

  /** Resolve `path` for opening on disk: absolute paths as-is; relative paths against top-level config dir. */
  std::string resolve_against_simulation_config(const std::string& path) const {
    namespace fs = std::filesystem;
    if (path.empty())
      return path;
    fs::path p(path);
    fs::path abs = p.is_absolute() ? fs::absolute(p)
                 : !config_file_path.empty()
                     ? fs::absolute(fs::path(config_file_path).parent_path() / p)
                     : fs::absolute(p);
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(abs, ec);
    return (ec ? abs : canon).string();
  }
};