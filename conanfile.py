from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import cross_building
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.env import VirtualBuildEnv, VirtualRunEnv
from conan.tools.files import apply_conandata_patches, collect_libs, copy, export_conandata_patches, get, rename, replace_in_file, rmdir, save
from conan.tools.microsoft import is_msvc, is_msvc_static_runtime
from conan.tools.scm import Version
import os
from fnmatch import fnmatch
import re
import textwrap


class LibPCLConan(ConanFile):
    python_requires = "camp_common/[>=0.1]@camposs/stable"

    name = "pcl"
    upstream_version = "1.13.1"
    package_revision = ""
    version = "{0}{1}".format(upstream_version, package_revision)

    url = "https://github.com/TUM-CONAN/conan-pcl"
    license = "BSD License"
    description = "The Point Cloud Library is for 2D/3D image and point cloud processing."

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
        "force_cuda_arch": ["ANY", ],
        "with_qt": [True, False],
    }

    default_options = {
        "shared": True,
        "fPIC": True,
        "with_cuda": True,
        "force_cuda_arch": "",
        "with_qt": False,
    }

    exports = [
        "select_compute_arch.cmake",
    ]

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        # PCL is not well prepared for c++ standard > 11...
        del self.settings.compiler.cppstd

        if self.settings.os == "Linux":
            self.options["boost"].fPIC = True

        if self.settings.os == "Windows":
            self.options["boost"].shared = True

    def requirements(self):
        if self.options.with_qt:
            self.requires("qt/6.5.0")
        self.requires("eigen/3.4.0")
        self.requires("boost/1.81.0")
        self.requires("flann/1.9.2", transitive_headers=True, transitive_libs=True)
        self.requires("zlib/1.2.13")

        if self.options.with_cuda:
            self.requires("cuda_dev_config/2.1@camposs/stable")

    def source(self):
        get(self,
            "https://github.com/PointCloudLibrary/pcl/archive/pcl-{0}.tar.gz".format(
                self.upstream_version),
            strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)

        def add_cmake_option(option, value):
            var_name = "{}".format(option).upper()
            value_str = "{}".format(value)
            var_value = "ON" if value_str == 'True' else "OFF" if value_str == 'False' else value_str
            tc.variables[var_name] = var_value

        for option, value in self.options.items():
            add_cmake_option(option, value)

        tc.variables["PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32"] = "ON"
        tc.variables["PCL_SHARED_LIBS"] = "ON" if self.options.shared else "OFF"

        tc.variables["WITH_PCAP"] = "OFF"
        tc.variables["WITH_DAVIDSDK"] = "OFF"
        tc.variables["WITH_ENSENSO"] = "OFF"
        tc.variables["WITH_OPENNI"] = "OFF"
        tc.variables["WITH_OPENNI2"] = "OFF"
        tc.variables["WITH_RSSDK"] = "OFF"
        tc.variables["WITH_QHULL"] = "OFF"

        tc.variables["WITH_QT"] = "ON" if self.options.with_qt else "OFF"

        if self.settings.os == "Windows":
            tc.variables["WITH_PNG"] = "OFF"

        tc.variables["BUILD_common"] = "ON"
        tc.variables["BUILD_2d"] = "ON"
        tc.variables["BUILD_features"] = "ON"
        tc.variables["BUILD_filters"] = "ON"
        tc.variables["BUILD_geometry"] = "ON"
        tc.variables["BUILD_io"] = "ON"
        tc.variables["BUILD_kdtree"] = "ON"
        tc.variables["BUILD_octree"] = "ON"
        tc.variables["BUILD_sample_consensus"] = "ON"
        tc.variables["BUILD_search"] = "ON"
        tc.variables["BUILD_ml"] = "ON"
        tc.variables["BUILD_segmentation"] = "ON"
        tc.variables["BUILD_registration"] = "ON"
        tc.variables["BUILD_surface"] = "ON"

        tc.variables["BUILD_apps"] = "OFF"
        tc.variables["BUILD_examples"] = "OFF"
        tc.variables["BUILD_tools"] = "OFF"
        tc.variables["BUILD_TESTS"] = "OFF"
        tc.variables["BUILD_simulation"] = "OFF"
        tc.variables["BUILD_visualization"] = "OFF"

        if self.options.with_cuda:
            tc.variables["BUILD_CUDA"] = "ON"
            tc.variables["BUILD_GPU"] = "ON"
            tc.variables["BUILD_gpu_containers"] = "ON"

            tc.variables["BUILD_gpu_kinfu"] = "OFF"
            tc.variables["BUILD_gpu_kinfu_large_scale"] = "OFF"
            # disabled due to incompatible use of thrust namespace with cuda sdk >= 11.6
            tc.variables["BUILD_cuda_sample_consensus"] = "OFF"
            tc.variables["BUILD_cuda_io"] = "OFF"
            tc.variables["BUILD_gpu_features"] = "OFF"
            tc.variables["BUILD_gpu_octree"] = "OFF"
            tc.variables["BUILD_gpu_surface"] = "OFF"
            if self.options.force_cuda_arch:
                forced_archs = filter(None, str(self.options.force_cuda_arch).split(","))
                tc.variables["PCL_FORCE_CUDA_ARCH"] = ";".join(forced_archs)
        else:
            tc.variables["BUILD_CUDA"] = "OFF"
            tc.variables["BUILD_GPU"] = "OFF"
            tc.variables["WITH_CUDA"] = "OFF"

        if self.settings.os == "Macos":
            tc.variables["BUILD_gpu_features"] = "OFF"

        #remove later
        tc.cache_variables["CMAKE_VERBOSE_MAKEFILE:BOOL"] = "ON"

        tc.generate()

        deps = CMakeDeps(self)
        deps.set_property("flann", "cmake_find_mode", "module")
        deps.set_property("flann", "cmake_file_name", "FLANN")
        deps.set_property("flann", "cmake_target_name", "FLANN::FLANN")
        deps.set_property("boost", "cmake_find_mode", "config")
        deps.set_property("boost", "cmake_file_name", "Boost")
        deps.set_property("boost", "cmake_target_name", "Boost::boost")
        deps.generate()

    def layout(self):
        cmake_layout(self, src_folder="source_folder")

    def build(self):

        if self.options.force_cuda_arch:
            tools.replace_in_file(os.path.join(source_subfolder, "cmake", "pcl_find_cuda.cmake"),
                """set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")""",
                """
                option(PCL_FORCE_CUDA_ARCH "Option to force CUDA Architectures to be built." "")
                if (PCL_FORCE_CUDA_ARCH)
                  set(__CUDA_ARCH_BIN ${PCL_FORCE_CUDA_ARCH})
                endif (PCL_FORCE_CUDA_ARCH)
                set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")
                """)
        if self.settings.os == "Windows":
            tools.replace_in_file(os.path.join(source_subfolder, "CMakeLists.txt"),
                """if("${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""",
                """if("${CONAN_SETTINGS_OS}" STREQUAL "Windows" OR "${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""")

        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = collect_libs(self)
        v_major, v_minor, v_micro = self.upstream_version.split(".")
        self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]
