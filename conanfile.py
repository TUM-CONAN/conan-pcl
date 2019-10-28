import os
import shutil

from conans import ConanFile, CMake, tools, AutoToolsBuildEnvironment
from conans.util import files
from fnmatch import fnmatch


class LibPCLConan(ConanFile):
    name = "pcl"
    upstream_version = "1.9.1"
    package_revision = ""
    version = "{0}{1}".format(upstream_version, package_revision)

    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "cuda": ["9.2", "10.0", "10.1", "None"]
    }
    default_options = [
        "shared=True",
        "cuda=None"
    ]
    default_options = tuple(default_options)
    exports = [
        "patches/CMakeProjectWrapper.txt",
        "patches/clang_macos.diff",
        "patches/kinfu.diff",
        "patches/pcl_cuda_thrust.diff",
        "patches/pcl_eigen.diff",
        "patches/pcl_gpu_error.diff"
    ]
    url = "https://git.ircad.fr/conan/conan-pcl"
    license = "BSD License"
    description = "The Point Cloud Library is a standalone, large scale, open project for 2D/3D image and point cloud processing."
    source_subfolder = "source_subfolder"
    build_subfolder = "build_subfolder"
    short_paths = True

    def configure(self):
        # del self.settings.compiler.libcxx
        # if 'CI' not in os.environ:
        #     os.environ["CONAN_SYSREQUIRES_MODE"] = "verify"
        self.options["Boost"].fPIC = True
        # self.options["Boost"].shared=True

    def requirements(self):
        self.requires("qt/5.12.2-r1@camposs/stable")
        self.requires("eigen/3.3.7@camposs/stable")
        self.requires("Boost/1.70.0@camposs/stable")
        self.requires("vtk/8.2.0-r1@camposs/stable")
        self.requires("openni/2.2.0-r3@camposs/stable")
        self.requires("flann/1.9.1-r2@camposs/stable")

        if tools.os_info.is_windows:
            self.requires("zlib/1.2.11@camposs/stable")

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
        tools.get("https://github.com/PointCloudLibrary/pcl/archive/pcl-{0}.tar.gz".format(self.upstream_version))
        os.rename("pcl-pcl-{0}".format(self.upstream_version), self.source_subfolder)

    def build(self):
        pcl_source_dir = os.path.join(self.source_folder, self.source_subfolder)
        shutil.move("patches/CMakeProjectWrapper.txt", "CMakeLists.txt")
        tools.patch(pcl_source_dir, "patches/clang_macos.diff")
        tools.patch(pcl_source_dir, "patches/kinfu.diff")
        tools.patch(pcl_source_dir, "patches/pcl_eigen.diff")
        tools.patch(pcl_source_dir, "patches/pcl_gpu_error.diff")
        tools.patch(pcl_source_dir, "patches/pcl_cuda_thrust.diff")

        # Use our own FindFLANN which take care of conan..
        os.remove(os.path.join(pcl_source_dir, 'cmake', 'Modules', 'FindFLANN.cmake'))

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

        if self.options.cuda != "None":
            cmake.definitions["BUILD_CUDA"] = "ON"
            cmake.definitions["BUILD_GPU"] = "ON"
            cmake.definitions["BUILD_gpu_kinfu"] = "ON"
            cmake.definitions["BUILD_gpu_kinfu_large_scale"] = "ON"
            cmake.definitions["BUILD_visualization"] = "ON"
            cmake.definitions["BUILD_surface"] = "ON"
            cmake.definitions["CUDA_ARCH_BIN"] = "3.0 3.5 5.0 5.2 6.1"

        if tools.os_info.is_macos:
            cmake.definitions["BUILD_gpu_features"] = "OFF"

        if tools.os_info.is_windows:
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "ON"
        else:
            # Clang >= 3.8 is not supported by CUDA 7.5
            cmake.definitions["CUDA_HOST_COMPILER"] = "/usr/bin/gcc"
            cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "OFF"

        if not tools.os_info.is_windows:
            cmake.definitions["CMAKE_POSITION_INDEPENDENT_CODE"] = "ON"

        cmake.configure(build_folder=self.build_subfolder)
        cmake.build()
        cmake.install()

    def cmake_fix_path(self, file_path, package_name):
        try:
            tools.replace_in_file(
                file_path,
                self.deps_cpp_info[package_name].rootpath.replace('\\', '/'),
                "${CONAN_" + package_name.upper() + "_ROOT}",
                strict=False
            )
        except:
            self.output.info("Ignoring {0}...".format(package_name))

    def package(self):
        for path, subdirs, names in os.walk(self.package_folder):
            for name in names:
                if fnmatch(name, '*.cmake'):
                    cmake_file = os.path.join(path, name)
                    
                    tools.replace_in_file(
                        cmake_file, 
                        self.package_folder.replace('\\', '/'), 
                        '${CONAN_PCL_ROOT}', 
                        strict=False
                    )
                    
                    self.cmake_fix_path(cmake_file, "boost")
                    self.cmake_fix_path(cmake_file, "eigen")
                    self.cmake_fix_path(cmake_file, "flann")
                    self.cmake_fix_path(cmake_file, "vtk")
                    self.cmake_fix_path(cmake_file, "openni")

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        v_major, v_minor, v_path = self.version.split(".")
        self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]
