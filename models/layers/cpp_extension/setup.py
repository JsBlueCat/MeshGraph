from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='meshgraph_cpp',
      ext_modules=[cpp_extension.CppExtension(
          'meshgraph_cpp', ['meshgraph.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
