import os

from fnmatch import fnmatch
from conans import ConanFile, CMake, tools


class LibPCLConan(ConanFile):
    name = "pcl"
    upstream_version = "1.11.1"
    package_revision = "-r1"
    version = "{0}{1}".format(upstream_version, package_revision)

    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
    }
    default_options = [
        "shared=True",
        "fPIC=True",
        "with_cuda=True"
    ]
    default_options = tuple(default_options)
    exports = [
        "select_compute_arch.cmake",
        # "patches/clang_macos.diff",
        # "patches/kinfu.diff",
        # "patches/pcl_eigen.diff",
        # "patches/pcl_gpu_error.diff",
        # "patches/point_cloud.diff",
        # "patches/pcl_supervoxel_clustering.diff",
        # "patches/cmake_add_new_boost_versions.diff",
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
        self.requires("ircad_common/1.0.3@camposs/stable")
        self.requires("qt/5.12.4-r4@camposs/stable")
        self.requires("eigen/3.3.9@camposs/stable")
        self.requires("Boost/1.75.0@camposs/stable")
        self.requires("flann/1.9.1-r6@camposs/stable")
        self.requires("zlib/1.2.11@camposs/stable")

        if self.options.with_cuda:
            self.requires("cuda_dev_config/[1.1]@camposs/stable")

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
        # tools.patch(pcl_source_dir, "patches/clang_macos.diff")
        # tools.patch(pcl_source_dir, "patches/kinfu.diff")
        # tools.patch(pcl_source_dir, "patches/pcl_eigen.diff")
        # tools.patch(pcl_source_dir, "patches/pcl_gpu_error.diff")
        # tools.patch(pcl_source_dir, "patches/point_cloud.diff")
        # tools.patch(pcl_source_dir, "patches/pcl_supervoxel_clustering.diff")
        # tools.patch(pcl_source_dir, "patches/cmake_add_new_boost_versions.diff")

        # patch for cuda arch >7.0

        # for path, subdirs, names in os.walk(pcl_source_dir,):
        #     for name in names:
        #         if fnmatch(name, "*.cu"):
        #             wildcard_file = os.path.join(path, name)

        #             # Fix package_folder paths
        #             tools.replace_in_file(
        #                 wildcard_file, "__all(", "__all_sync(0xFFFFFFFF,", strict=False)
        #             tools.replace_in_file(
        #                 wildcard_file, "__any(", "__any_sync(0xFFFFFFFF,", strict=False)
        #             tools.replace_in_file(
        #                 wildcard_file, "__ballot(",
        #                 "__ballot_sync(0xFFFFFFFF,", strict=False)

        # Use our own FindFLANN which take care of conan..
        # os.remove(
        #     os.path.join(
        #         pcl_source_dir,
        #         'cmake',
        #         'Modules',
        #         'FindFLANN.cmake'))

        tools.replace_in_file(os.path.join(self.source_subfolder, "common", "include", "pcl", "types.h"),
            """
  using uint8_t PCL_DEPRECATED(1, 12, "use std::uint8_t instead of pcl::uint8_t") = std::uint8_t;
  using int8_t PCL_DEPRECATED(1, 12, "use std::int8_t instead of pcl::int8_t") = std::int8_t;
  using uint16_t PCL_DEPRECATED(1, 12, "use std::uint16_t instead of pcl::uint16_t") = std::uint16_t;
  using int16_t PCL_DEPRECATED(1, 12, "use std::uint16_t instead of pcl::int16_t") = std::int16_t;
  using uint32_t PCL_DEPRECATED(1, 12, "use std::uint32_t instead of pcl::uint32_t") = std::uint32_t;
  using int32_t PCL_DEPRECATED(1, 12, "use std::int32_t instead of pcl::int32_t") = std::int32_t;
  using uint64_t PCL_DEPRECATED(1, 12, "use std::uint64_t instead of pcl::uint64_t") = std::uint64_t;
  using int64_t PCL_DEPRECATED(1, 12, "use std::int64_t instead of pcl::int64_t") = std::int64_t;
  using int_fast16_t PCL_DEPRECATED(1, 12, "use std::int_fast16_t instead of pcl::int_fast16_t") = std::int_fast16_t;
""",
            """
  using uint8_t = std::uint8_t;
  using int8_t = std::int8_t;
  using uint16_t = std::uint16_t;
  using int16_t = std::int16_t;
  using uint32_t = std::uint32_t;
  using int32_t = std::int32_t;
  using uint64_t = std::uint64_t;
  using int64_t = std::int64_t;
  using int_fast16_t = std::int_fast16_t;
""")

        # Import common flags and defines
        import common

        # Generate Cmake wrapper
        common.generate_cmake_wrapper(
            cmakelists_path='CMakeLists.txt',
            source_subfolder=self.source_subfolder,
            build_type=self.settings.build_type,
            setup_cuda=True
        )

        cmake = CMake(self)

        cmake.definitions["BUILD_apps"] = "OFF"
        cmake.definitions["BUILD_examples"] = "OFF"
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
        cmake.definitions["BUILD_tools"] = "OFF"
        cmake.definitions["PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32"] = "ON"
        cmake.definitions["PCL_SHARED_LIBS"] = "ON"
        cmake.definitions["WITH_PCAP"] = "OFF"
        cmake.definitions["WITH_DAVIDSDK"] = "OFF"
        cmake.definitions["WITH_ENSENSO"] = "OFF"
        cmake.definitions["WITH_OPENNI"] = "OFF"
        cmake.definitions["WITH_OPENNI2"] = "OFF"
        cmake.definitions["WITH_RSSDK"] = "OFF"
        cmake.definitions["WITH_QHULL"] = "OFF"
        cmake.definitions["BUILD_TESTS"] = "OFF"
        cmake.definitions["BUILD_ml"] = "ON"
        cmake.definitions["BUILD_simulation"] = "OFF"
        cmake.definitions["BUILD_segmentation"] = "ON"
        cmake.definitions["BUILD_registration"] = "ON"

        if self.options.with_cuda:
            cmake.definitions["BUILD_CUDA"] = "ON"
            cmake.definitions["BUILD_GPU"] = "ON"
            cmake.definitions["BUILD_gpu_containers"] = "ON"
            cmake.definitions["BUILD_gpu_kinfu"] = "OFF"
            cmake.definitions["BUILD_gpu_kinfu_large_scale"] = "OFF"
            cmake.definitions["BUILD_visualization"] = "OFF"
            cmake.definitions["BUILD_surface"] = "ON"
            cmake.definitions["CUDA_ARCH_BIN"] = ' '.join(
                common.get_cuda_arch())

        if tools.os_info.is_macos:
            cmake.definitions["BUILD_gpu_features"] = "OFF"

        if tools.os_info.is_windows:
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "ON"
        else:
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "OFF"

        cmake.configure()
        cmake.build()
        cmake.install()

    def package(self):
        # Import common flags and defines
        import common

        common.fix_conan_path(self, self.package_folder, '*.cmake')

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        v_major, v_minor, v_path = self.version.split(".")
        self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]
