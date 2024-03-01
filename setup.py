import os
import platform
import subprocess
import sys
import sysconfig

import numpy as np
from packaging.version import Version, parse
from setuptools import Extension, setup

_BUILD_ATTEMPTS = 0
os.environ['CC']='icpx'
os.environ['CXX']='icpx'

# This is copied from @robbuckley's fix for Panda's
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behavior which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.pcuda-comp-generalizey
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system: Version = parse(platform.mac_ver()[0])
        python_target: Version = parse(sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < Version('10.9') and current_system >= Version('10.9'):
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'


def find_in_path(name, path):
    """Find a file in a search path and return its full path."""
    # adapted from:
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def get_cuda_path():
    """Return a tuple with (base_cuda_directory, full_path_to_nvcc_compiler)."""
    # Inspired by https://github.com/benfred/implicit/blob/master/cuda_setup.py
    nvcc_bin = "nvcc.exe" if sys.platform == "win32" else "nvcc"

    if "CUDAHOME" in os.environ:
        cuda_home = os.environ["CUDAHOME"]
    elif "CUDA_PATH" in os.environ:
        cuda_home = os.environ["CUDA_PATH"]
    else:
        # otherwise, search the PATH for NVCC
        found_nvcc = find_in_path(nvcc_bin, os.environ["PATH"])
        if found_nvcc is None:
            print(
                "The nvcc binary could not be located in your $PATH. Either "
                "add it to your path, or set $CUDAHOME to enable CUDA.",
            )
            return None
        cuda_home = os.path.dirname(os.path.dirname(found_nvcc))
    if not os.path.exists(os.path.join(cuda_home, "include")):
        print("Failed to find cuda include directory, using /usr/local/cuda")
        cuda_home = "/usr/local/cuda"

    nvcc = os.path.join(cuda_home, "bin", nvcc_bin)
    if not os.path.exists(nvcc):
        print("Failed to find nvcc compiler in %s, trying /usr/local/cuda" % nvcc)
        cuda_home = "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", nvcc_bin)

    return cuda_home, nvcc

def get_intel_path():
    """Return a tuple with (base_intel_directory, full_path_to_icpx_compiler)"""
    # eventually remove hard-code paths
    relative_path_to_icpx = "compiler/latest/linux/bin/icpx"
    if "ONEAPI_ROOT" in os.environ:
        oneapi_root = os.environ["ONEAPI_ROOT"]
        icpx = os.path.join(oneapi_root, relative_path_to_icpx)
        return oneapi_root, icpx
    else:
        print('Failed to find OneAPI root. Be sure setvars.sh has been executed \
              in your shell and ONEAPI_ROOT exists in your environment.')
        return None

def compile_cuda_module(host_args):
    libname = '_cext_gpu.lib' if sys.platform == 'win32' else 'lib_cext_gpu.a'
    lib_out = 'build/' + libname
    if not os.path.exists('build/'):
        os.makedirs('build/')

    _, nvcc = get_cuda_path()

    print("NVCC ==> ", nvcc)
    arch_flags = (
        "-arch=sm_37 "
        "-gencode=arch=compute_37,code=sm_37 "
        "-gencode=arch=compute_70,code=sm_70 "
        "-gencode=arch=compute_75,code=sm_75 "
        "-gencode=arch=compute_75,code=compute_75"
    )
    nvcc_command = (
        f"-allow-unsupported-compiler shap/cext/_cext_gpu.cu -lib -o {lib_out} "
        f"-Xcompiler {','.join(host_args)} "
        f"--include-path {sysconfig.get_path('include')} "
        "--std c++14 "
        "--expt-extended-lambda "
        f"--expt-relaxed-constexpr {arch_flags}"
    )
    print("Compiling cuda extension, calling nvcc with arguments:")
    print([nvcc] + nvcc_command.split(' '))
    subprocess.run([nvcc] + nvcc_command.split(' '), check=True)
    return 'build', '_cext_gpu'

def compile_sycl_module(host_args, icpx):
    libname = '_cext_gpu.lib' if sys.platform == 'win32' else 'lib_cext_gpu.a'
    lib_out = 'build/' + libname
    if not os.path.exists('build/'):
        os.makedirs('build/')
    
    print("ICPX ==> ", icpx)
    # Do have any arch_flags to set?
    '''
    arch_flags = (
        "-arch=sm_37 "
        "-gencode=arch=compute_37,code=sm_37 "
        "-gencode=arch=compute_70,code=sm_70 "
        "-gencode=arch=compute_75,code=sm_75 "
        "-gencode=arch=compute_75,code=compute_75"
    )
    '''
    icpx_command = (
        f"shap/cext/dpct_output/_cext_gpu.dp.cpp -c -o {lib_out} "
        f"-I {sysconfig.get_path('include')} "
        #f"-stdlib=c++14 "
        f"{' '.join(host_args)}"
    )
    print([icpx] + icpx_command.split(' '))
    subprocess.run([icpx] + icpx_command.split(' '), check=True )
    return 'build', '_cext_gpu'


def run_setup(*, with_binary, with_cuda, with_sycl):
    ext_modules = []
    if with_binary:
        compile_args = []
        if sys.platform == 'zos':
            compile_args.append('-qlonglong')
        if sys.platform == 'win32':
            compile_args.append('/MD')

        compile_args.append('-fsycl')
        ext_modules.append(
            Extension('shap._cext', sources=['shap/cext/_cext.cc'],
                      include_dirs=[np.get_include()],
                      extra_compile_args=compile_args))
    if with_cuda:
        try:
            cuda_home, _ = get_cuda_path()
            if sys.platform == 'win32':
                cudart_path = cuda_home + '/lib/x64'
            else:
                cudart_path = cuda_home + '/lib64'
                compile_args.append('-fPIC')

            lib_dir, lib = compile_cuda_module(compile_args)

            ext_modules.append(
                Extension('shap._cext_gpu', sources=['shap/cext/_cext_gpu.cc'],
                          extra_compile_args=compile_args,
                          include_dirs=[np.get_include()],
                          library_dirs=[lib_dir, cudart_path],
                          libraries=[lib, 'cudart'],
                          depends=['shap/cext/_cext_gpu.cu', 'shap/cext/gpu_treeshap.h', 'setup.py'])
            )
        except Exception as e:
            raise Exception("Error building cuda module: " + repr(e)) from e

    if with_sycl:
        try:
            intel_home, icpx = get_intel_path()
            compile_args.append('-fPIC')
            lib_dir, lib = compile_sycl_module(compile_args, icpx)
            intelart_path = os.path.join(intel_home, 'compiler/latest/linux/bin/intel64') 
            ext_modules.append(
                #Extension('shap.dpct_output._cext_gpu', sources=['shap/cext/_cext_gpu.cc'],
                Extension('shap._cext_gpu', sources=['shap/cext/_cext_gpu.cc'],
                          extra_compile_args=compile_args,
                          include_dirs=[np.get_include()],
                          library_dirs=[lib_dir, intelart_path],
                          libraries=[lib],
                          depends=['shap/cext/dpct_output/_cext_gpu.dp.cpp',
                                   'shap/cext/dpct_output/gpu_treeshap.h',
                                   'setup.py'])
            )
        except Exception as e:
            raise Exception("Error building sycl module: " + repr(e)) from e


    setup(ext_modules=ext_modules)


def try_run_setup(*, with_binary, with_cuda, with_sycl):
    """ Fails gracefully when various install steps don't work.
    """
    global _BUILD_ATTEMPTS
    _BUILD_ATTEMPTS += 1

    try:
        print(f"Attempting to build SHAP: {with_binary=}, {with_cuda=}, {with_sycl=} (Attempt {_BUILD_ATTEMPTS})")
        run_setup(with_binary=with_binary, with_cuda=with_cuda, with_sycl=with_sycl)
    except Exception as e:
        print("Exception occurred during setup,", str(e))

        if with_cuda:
            with_cuda = False
            print("WARNING: Could not compile cuda extensions.")
            print("Retrying SHAP build without cuda extension...")
        elif with_binary:
            with_binary = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            print("Retrying SHAP build without binary extension...")
        elif with_sycl:
            with_sycl = False
            print("WARNING: Could not compile SYCL implementation of GPUTree.")
            print("Retrying SHAP build without SYCL extensions...")
        else:
            print("ERROR: Failed to build!")
            raise

        try_run_setup(with_binary=with_binary, with_cuda=with_cuda, with_sycl=with_sycl)


# we seem to need this import guard for appveyor
if __name__ == "__main__":
    try_run_setup(with_binary=True, with_cuda=True, with_sycl=True)
