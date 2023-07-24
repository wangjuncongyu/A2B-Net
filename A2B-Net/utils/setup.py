import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.extension import Extension

def find_in_path(name,path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc.exe', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib', 'x64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    CppExtension(
        "rotation.riou",
        ["rotation/riou.pyx"],
        extra_compile_args={'cxx': []},
        include_dirs=[numpy_include]
    ),
    CppExtension(
        "rotation.rotate_cython_nms",
        ["rotation/rotate_cython_nms.pyx"],
        extra_compile_args={'cxx': []},
        include_dirs=[numpy_include]
    ),

    CppExtension(
        "rotation.rotate_circle_nms",
        ["rotation/rotate_circle_nms.pyx"],
        extra_compile_args={'cxx': []},
        include_dirs=[numpy_include]
    ),
    
    
    CUDAExtension('rotation.rbbox_overlaps',
              ['rotation/rbbox_overlaps.cpp', 'rotation/rbbox_overlaps_kernel.cu'],
              library_dirs=[CUDA['lib64']],
              libraries=['cudart'],
              language='c++',
              # runtime_library_dirs=[CUDA['lib64']],
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with nvcc and not with
              # gcc the implementation of this trick is in customize_compiler() below
              extra_compile_args={'cxx':[],
                                  'nvcc': ['-arch=sm_75',
                                           '--ptxas-options=-v',
                                           '-c',
                                           '--compiler-options',
                                           ]},
              include_dirs=[numpy_include, CUDA['include']]
              ),
    CUDAExtension('rotation.rotate_polygon_nms',
              ['rotation/rotate_polygon_nms_kernel.cu', 'rotation/rotate_polygon_nms.cpp'],
              library_dirs=[CUDA['lib64']],
              libraries=['cudart'],
              language='c++',
              # runtime_library_dirs=[CUDA['lib64']],
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with nvcc and not with
              # gcc the implementation of this trick is in customize_compiler() below
              extra_compile_args={'cxx':[],#'gcc': ["-Wno-unused-function"],
                                  'nvcc': ['-arch=sm_75',
                                           '--ptxas-options=-v',
                                           '-c',
                                           '--compiler-options',
                                           ]},
              include_dirs=[numpy_include, CUDA['include']]
              ),
]

setup(
    name='RRPN',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': BuildExtension},
)
