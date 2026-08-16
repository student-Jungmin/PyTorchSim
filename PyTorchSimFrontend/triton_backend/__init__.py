"""The `npu` codegen route: Inductor's Triton backend + the tnpu passes.

    Inductor -> TritonNPUScheduling.define_kernel   (scheduling.py)
             -> triton_npu_compile                  (codecache.py, kernel_spec.py)
             -> triton-npu, in a subprocess         (tnpu_bridge.py)
             -> Spike (functional.py) / TOGSim (timing.py)

Registered for `npu` at device import; see PyTorchSimDevice/torch_openreg.
"""

from . import _triton_compat, inductor_templates

_triton_compat.install()
inductor_templates.install()

from .scheduling import TritonNPUScheduling
from .wrapper_codegen import TritonNPUWrapperCodegen
