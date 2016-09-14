from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os, numpy

setup(
    ext_modules = cythonize([
    Extension("siggen", ["siggen.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["MGDOmjd_siggen"],
              library_dirs = [os.getcwd()])
    ])
)