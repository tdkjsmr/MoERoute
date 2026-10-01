"""构建独立路由扩展；在已有 PyTorch 环境的 CUDA 服务器上执行。"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="moe-router",
    version="0.1.0",
    py_modules=["router"],
    ext_modules=[
        CUDAExtension(
            name="_moe_router",
            # 相对路径满足 setuptools 的源码清单要求。
            sources=["routing.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "-gencode=arch=compute_86,code=sm_86",
                    "--generate-line-info",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
