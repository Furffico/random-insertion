from setuptools import Extension, setup
import glob, os
import numpy

numpy_header_path = os.path.join(
    os.path.dirname(numpy.__file__), 
    '_core' if numpy.__version__>='2.0' else 'core', 
    'include'
)

_core_ext = Extension(
    name = "random_insertion._core",
    sources = glob.glob("random_insertion/src/*.cpp"),
    include_dirs = ["random_insertion/src", numpy_header_path],
    extra_compile_args = ["-std=c++17", "-O3", "-Os"],
    language = "c++",
)

setup(
    ext_modules=[_core_ext]
)