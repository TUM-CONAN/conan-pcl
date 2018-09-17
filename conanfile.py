from conans import ConanFile, CMake, tools, AutoToolsBuildEnvironment
from conans.util import files
import os
import shutil

class LibPCLConan(ConanFile):
    name = "pcl"
    version = "1.8.1-rev-9dae1ea"
    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "use_cuda": [True, False]
    }
    default_options = [
        "shared=True",
        "use_cuda=False"
    ]
    default_options = tuple(default_options)
    exports = [
        "patches/CMakeProjectWrapper.txt",
        "patches/clang_macos.diff",
        "patches/kinfu.diff",
        "patches/pcl_eigen.diff",
        "patches/pcl_gpu_error.diff"
    ]
    url = "https://gitlab.lan.local/conan/conan-pcl"
    license="BSD License"
    description = "The Point Cloud Library is a standalone, large scale, open project for 2D/3D image and point cloud processing."
    source_subfolder = "source_subfolder"
    build_subfolder = "build_subfolder"
    short_paths = True

    def requirements(self):
        self.requires("qt/5.11.1@fw4spl/stable")
        self.requires("eigen/3.3.4@fw4spl/stable")
        self.requires("boost/1.67.0@fw4spl/stable")
        self.requires("qt/5.11.1@fw4spl/stable")
        self.requires("vtk/8.0.1@fw4spl/stable")
        self.requires("openni/2.2.0-rev-958951f@fw4spl/stable")
        self.requires("flann/1.9.1@fw4spl/stable")

        if not tools.os_info.is_linux:
            self.requires("zlib/1.2.11@fw4spl/stable")

    def system_requirements(self):
        if tools.os_info.linux_distro == "ubuntu":
            pack_names = [
                "zlib1g-dev"
            ]
            installer = tools.SystemPackageTool()
            installer.update()
            installer.install(" ".join(pack_names))

    def source(self):
        rev = "9dae1eaa6750932db23d157cd624ef61ccd5544f"
        tools.get("https://github.com/PointCloudLibrary/pcl/archive/{0}.tar.gz".format(rev))
        os.rename("pcl-" + rev, self.source_subfolder)

    def build(self):
        pcl_source_dir = os.path.join(self.source_folder, self.source_subfolder)
        shutil.move("patches/CMakeProjectWrapper.txt", "CMakeLists.txt")
        tools.patch(pcl_source_dir, "patches/clang_macos.diff")
        tools.patch(pcl_source_dir, "patches/kinfu.diff")
        tools.patch(pcl_source_dir, "patches/pcl_eigen.diff")
        tools.patch(pcl_source_dir, "patches/pcl_gpu_error.diff")

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
        cmake.definitions["WITH_RSSDK"] = "OFF"
        cmake.definitions["WITH_QHULL"] = "OFF"       
        cmake.definitions["BUILD_TESTS"] = "OFF"
        cmake.definitions["BUILD_ml"] = "ON"
        cmake.definitions["BUILD_simulation"] = "OFF"
        cmake.definitions["BUILD_segmentation"] = "ON"
        cmake.definitions["BUILD_registration"] = "ON"
        #cmake.definitions["VTK_DIR"] = self.deps_cpp_info["vtk"].lib_paths[0].replace('\\', '/')
        #cmake.definitions["ZLIB_INCLUDE_DIR"] = self.deps_cpp_info["zlib"].include_paths[0].replace('\\', '/')
        #cmake.definitions["CUDA_TOOLKIT_ROOT_DIR"] = ${CUDA_TOOLKIT_ROOT_DIR}
        #cmake.definitions["BOOST_ROOT"] = ${CMAKE_INSTALL_PREFIX}
        #cmake.definitions["FLANN_INCLUDE_DIR"] = ${CMAKE_INSTALL_PREFIX}/include/flann
        #cmake.definitions["PCL_ENABLE_SSE"] = ${ENABLE_SSE_SUPPORT}
        #cmake.definitions["OPENNI2_INCLUDE_DIRS"] = ${CMAKE_INSTALL_PREFIX}/include/openni2
        #cmake.definitions["QT_QMAKE_EXECUTABLE"] = ${CMAKE_INSTALL_PREFIX}/bin/qmake


        if self.options.use_cuda:
            cmake.definitions["BUILD_CUDA"] = "ON"
            cmake.definitions["BUILD_GPU"] = "ON"
            cmake.definitions["BUILD_gpu_kinfu"] = "ON"
            cmake.definitions["BUILD_gpu_kinfu_large_scale"] = "ON"
            cmake.definitions["BUILD_visualization"] = "ON"
            cmake.definitions["BUILD_surface"] = "ON"

        if tools.os_info.is_macos:
            cmake.definitions["BUILD_gpu_features"] = "OFF"

        if tools.os_info.is_windows:
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "ON"
        else:
            # Clang >= 3.8 is not supported by CUDA 7.5
            cmake.definitions["CUDA_HOST_COMPILER"] = "/usr/bin/gcc"
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "OFF"

        cmake.configure(build_folder=self.build_subfolder)
        cmake.build()
        cmake.install()
        cmake.patch_config_paths()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
