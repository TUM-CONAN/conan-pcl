import os

from fnmatch import fnmatch
from conans import ConanFile, CMake, tools


class LibPCLConan(ConanFile):
    python_requires = "camp_common/[>=0.1]@camposs/stable"

    name = "pcl"
    upstream_version = "1.12.1"
    package_revision = "-r1"
    version = "{0}{1}".format(upstream_version, package_revision)

    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
        "force_cuda_arch": "ANY",
        "with_qt": [True, False],
    }

    default_options = {
        "shared":True,
        "fPIC":True,
        "with_cuda":True,
        "force_cuda_arch":"",
        "with_qt":False,
    }

    exports = [
        "select_compute_arch.cmake",
    ]
    url = "https://git.ircad.fr/conan/conan-pcl"
    license = "BSD License"
    description = "The Point Cloud Library is for 2D/3D image and point cloud processing."
    source_subfolder = "source_subfolder"
    short_paths = True

    def config_options(self):
        if tools.os_info.is_windows:
            del self.options.fPIC

    def configure(self):
        # PCL is not well prepared for c++ standard > 11...
        del self.settings.compiler.cppstd

        # if 'CI' not in os.environ:
        #     os.environ["CONAN_SYSREQUIRES_MODE"] = "verify"

        if self.settings.os == "Linux":
            self.options["Boost"].fPIC = True

        if tools.os_info.is_windows:
            self.options["Boost"].shared=True


    def requirements(self):
        if self.options.with_qt:
            self.requires("qt/5.15.4@camposs/stable")
        self.requires("eigen/3.3.9-r1@camposs/stable")
        self.requires("Boost/1.79.0@camposs/stable")
        self.requires("flann/1.9.1-r7@camposs/stable")
        self.requires("zlib/1.2.12")

        if self.options.with_cuda:
            self.requires("cuda_dev_config/2.0@camposs/stable")

    def build_requirements(self):
        if tools.os_info.linux_distro == "linuxmint":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g-dev")
        if tools.os_info.linux_distro == "ubuntu":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g-dev")

    def system_requirements(self):
        if tools.os_info.linux_distro == "linuxmint":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g")
        if tools.os_info.linux_distro == "ubuntu":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g")

    def source(self):
        tools.get(
            "https://github.com/PointCloudLibrary/pcl/archive/pcl-{0}.tar.gz".format(
                self.upstream_version))
        os.rename(
            "pcl-pcl-{0}".format(self.upstream_version),
            self.source_subfolder)

    def build(self):
        pcl_source_dir = os.path.join(
            self.source_folder, self.source_subfolder)

        if self.options.force_cuda_arch:
            tools.replace_in_file(os.path.join(pcl_source_dir, "cmake", "pcl_find_cuda.cmake"),
                """set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")""",
                """
                option(PCL_FORCE_CUDA_ARCH "Option to force CUDA Architectures to be built." "")
                if (PCL_FORCE_CUDA_ARCH)
                  set(__CUDA_ARCH_BIN ${PCL_FORCE_CUDA_ARCH})
                endif (PCL_FORCE_CUDA_ARCH)
                set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")
                """)
        if tools.os_info.is_windows:
            tools.replace_in_file(os.path.join(pcl_source_dir, "CMakeLists.txt"),
                """if("${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""",
                """if("${CONAN_SETTINGS_OS}" STREQUAL "Windows" OR "${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""")

        # Import common flags and defines
        common = self.python_requires["camp_common"].module

        # Generate Cmake wrapper
        common.generate_cmake_wrapper(
            cmakelists_path='CMakeLists.txt',
            source_subfolder=self.source_subfolder,
            build_type=self.settings.build_type,
            create_minimal_wrapper=True,
        )

        cmake = CMake(self)

        cmake.definitions["PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32"] = "ON"
        cmake.definitions["PCL_SHARED_LIBS"] = "ON" if self.options.shared else "OFF"

        cmake.definitions["WITH_PCAP"] = "OFF"
        cmake.definitions["WITH_DAVIDSDK"] = "OFF"
        cmake.definitions["WITH_ENSENSO"] = "OFF"
        cmake.definitions["WITH_OPENNI"] = "OFF"
        cmake.definitions["WITH_OPENNI2"] = "OFF"
        cmake.definitions["WITH_RSSDK"] = "OFF"
        cmake.definitions["WITH_QHULL"] = "OFF"

        cmake.definitions["WITH_QT"] = "ON" if self.options.with_qt else "OFF"

        if tools.os_info.is_windows:
            cmake.definitions["WITH_PNG"] = "OFF"

        cmake.definitions["BUILD_common"] = "ON"
        cmake.definitions["BUILD_2d"] = "ON"
        cmake.definitions["BUILD_features"] = "ON"
        cmake.definitions["BUILD_filters"] = "ON"
        cmake.definitions["BUILD_geometry"] = "ON"
        cmake.definitions["BUILD_io"] = "ON"
        cmake.definitions["BUILD_kdtree"] = "ON"
        cmake.definitions["BUILD_octree"] = "ON"
        cmake.definitions["BUILD_sample_consensus"] = "ON"
        cmake.definitions["BUILD_search"] = "ON"
        cmake.definitions["BUILD_ml"] = "ON"
        cmake.definitions["BUILD_segmentation"] = "ON"
        cmake.definitions["BUILD_registration"] = "ON"
        cmake.definitions["BUILD_surface"] = "ON"

        cmake.definitions["BUILD_apps"] = "OFF"
        cmake.definitions["BUILD_examples"] = "OFF"
        cmake.definitions["BUILD_tools"] = "OFF"
        cmake.definitions["BUILD_TESTS"] = "OFF"
        cmake.definitions["BUILD_simulation"] = "OFF"
        cmake.definitions["BUILD_visualization"] = "OFF"

        if self.options.with_cuda:
            cmake.definitions["BUILD_CUDA"] = "ON"
            cmake.definitions["BUILD_GPU"] = "ON"
            cmake.definitions["BUILD_gpu_containers"] = "ON"

            cmake.definitions["BUILD_gpu_kinfu"] = "OFF"
            cmake.definitions["BUILD_gpu_kinfu_large_scale"] = "OFF"
            # disabled due to incompatible use of thrust namespace with cuda sdk >= 11.6
            cmake.definitions["BUILD_cuda_sample_consensus"] = "OFF"
            cmake.definitions["BUILD_cuda_io"] = "OFF"
            cmake.definitions["BUILD_gpu_features"] = "OFF"
            cmake.definitions["BUILD_gpu_octree"] = "OFF"
            cmake.definitions["BUILD_gpu_surface"] = "OFF"

            if self.options.force_cuda_arch:
                forced_archs = filter(None, str(self.options.force_cuda_arch).split(","))
                cmake.definitions["PCL_FORCE_CUDA_ARCH"] = ";".join(forced_archs)
        else:
            cmake.definitions["BUILD_CUDA"] = "OFF"
            cmake.definitions["BUILD_GPU"] = "OFF"
            cmake.definitions["WITH_CUDA"] = "OFF"

        if tools.os_info.is_macos:
            cmake.definitions["BUILD_gpu_features"] = "OFF"

        # if tools.os_info.is_windows:
        #     cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "ON"
        # else:
            # cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "OFF"

        cmake.configure()
        cmake.build()
        cmake.install()

    def package(self):
        # Import common flags and defines
        common = self.python_requires["camp_common"].module

        common.fix_conan_path(self, self.package_folder, '*.cmake')

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        v_major, v_minor, v_micro = self.upstream_version.split(".")
        self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]
