"""Wrapper codegen for the Triton route.

Reuses ExtensionWrapperCodegen wholesale and adds the one import the generated
module needs: `triton_npu_compile`.
"""
from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen

from . import codecache


class TritonNPUWrapperCodegen(ExtensionWrapperCodegen):
    def wrap_kernel_call(self, name, call_args):
        """Render the call args before joining them.

        `generate` hands call_args straight to a join, so a sympy Integer is a
        TypeError. Every call passes here, which is what makes it one place.
        """
        return super().wrap_kernel_call(
            name, self.prepare_triton_kernel_call(call_args))

    def write_header(self):
        super().write_header()
        self.header.splice(
            f"""
            from {codecache.__name__} import triton_npu_compile
            """
        )
