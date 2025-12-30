import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Debug" if self.debug else "Release"

        # Try to import pybind11 to get the cmake prefix path
        try:
            import pybind11
            pybind11_cmake_path = pybind11.get_cmake_dir()
        except ImportError:
            # Should not happen if pyproject.toml is correct
            pybind11_cmake_path = ""

        # Find cuQuantum root and specific paths
        try:
            import cuquantum
            import site
            # Search site-packages for custatevec.h and libcustatevec.so
            site_packages = site.getsitepackages()[0]
            
            custatevec_include = ""
            custatevec_lib = ""
            
            # Walk through site-packages to find header and lib
            # This is brute-force but reliable given unclear wheel structure in build env
            custatevec_lib_path = ""
            for root, dirs, files in os.walk(site_packages):
                if "custatevec.h" in files:
                    custatevec_include = root
                    print(f"Found header at: {custatevec_include}")
                
                # Look for libcustatevec.so or versioned .so
                for f in files:
                    if f.startswith("libcustatevec.so"):
                        custatevec_lib = root
                        custatevec_lib_path = os.path.join(root, f)
                        print(f"Found lib at: {custatevec_lib_path}")
                        break
                
                if custatevec_include and custatevec_lib_path:
                    break
                    
            if not custatevec_include:
                # Fallback to cuquantum module path
                cuquantum_root = os.path.dirname(cuquantum.__file__)
                if os.path.exists(os.path.join(cuquantum_root, "include", "custatevec.h")):
                    custatevec_include = os.path.join(cuquantum_root, "include")
            
            if not custatevec_lib_path:
                cuquantum_root = os.path.dirname(cuquantum.__file__)
                lib_dir = os.path.join(cuquantum_root, "lib")
                if os.path.exists(lib_dir):
                     for f in os.listdir(lib_dir):
                         if f.startswith("libcustatevec.so"):
                             custatevec_lib_path = os.path.join(lib_dir, f)
                             break
            
            # Find nvcc in site-packages or sys.prefix
            nvcc_path = ""
            search_roots = site.getsitepackages() + [sys.prefix, os.path.dirname(sys.executable)]
            print(f"Searching for nvcc in: {search_roots}")
            
            for search_root in search_roots:
                # Avoid permission denied errors or deep recursion
                for root, dirs, files in os.walk(search_root):
                    if "nvcc" in files:
                        nvcc_candidate = os.path.join(root, "nvcc")
                        if os.access(nvcc_candidate, os.X_OK) and "bin" in root:
                            nvcc_path = nvcc_candidate
                            print(f"Found nvcc at: {nvcc_path}")
                            break
                if nvcc_path:
                    break

        except ImportError:
            custatevec_include = ""
            custatevec_lib_path = ""
            nvcc_path = ""

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={pybind11_cmake_path}",
            f"-DCUSTATEVEC_INCLUDE_DIR={custatevec_include}",
            f"-DCUSTATEVEC_LIB_PATH={custatevec_lib_path}",
        ]
        
        if nvcc_path:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
        else:
             print("WARNING: nvcc not found in python environment")

        # Helper to get package root
        def get_pkg_root(module):
            if hasattr(module, "__file__") and module.__file__:
                return os.path.dirname(module.__file__)
            elif hasattr(module, "__path__"):
                # Namespace package or directory
                return list(module.__path__)[0]
            return None

        # Find nvidia-cuda-runtime and cublas paths to fix linking against system libs
        try:
            import nvidia.cuda_runtime
            cudart_root = get_pkg_root(nvidia.cuda_runtime)
            if cudart_root:
                cmake_args.append(f"-DPYTHON_CUDART_ROOT={cudart_root}")
        except ImportError:
            pass
            
        try:
            import nvidia.cublas
            cublas_root = get_pkg_root(nvidia.cublas)
            if cublas_root:
                cmake_args.append(f"-DPYTHON_CUBLAS_ROOT={cublas_root}")
        except ImportError:
            pass

        try:
            # Need nvcc include for crt/host_defines.h
            import nvidia.cuda_nvcc
            nvcc_root = get_pkg_root(nvidia.cuda_nvcc)
            if nvcc_root:
                cmake_args.append(f"-DPYTHON_NVCC_ROOT={nvcc_root}")
        except ImportError:
            pass

        build_args = ["--config", cfg, "-j"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True
        )


setup(
    name="torchqml",
    version="0.2.0",
    author="Your Name",
    description="PyTorch Quantum ML with cuQuantum C++ Backend",
    packages=find_packages(),
    ext_modules=[CMakeExtension("torchqml._C")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "python-backend": ["cupy-cuda12x", "cuquantum-python-cu12"],
    },
)
