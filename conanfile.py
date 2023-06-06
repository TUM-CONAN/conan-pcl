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
            self.requires("qt/6.4.2")
        self.requires("eigen/3.4.0")
        self.requires("boost/1.81.0")
        self.requires("flann/1.9.2", transitive_headers=True, transitive_libs=True)
        # self.requires("lz4/1.9.4")

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
        tc.variables["WITH_VTK"] = "OFF"

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
        deps.set_property("flann", "cmake_find_mode", "config")
        deps.set_property("flann", "cmake_file_name", "FLANN")
        deps.set_property("flann", "cmake_target_name", "FLANN::FLANN")

        deps.set_property("boost", "cmake_find_mode", "config")
        deps.set_property("boost", "cmake_file_name", "Boost")
        deps.set_property("boost", "cmake_target_name", "Boost")

        deps.set_property("eigen", "cmake_find_mode", "module")
        deps.set_property("eigen", "cmake_file_name", "Eigen")
        deps.set_property("eigen", "cmake_target_name", "eigen")
        deps.generate()

    def layout(self):
        cmake_layout(self, src_folder="source_folder")

    def build(self):

        # problem with transitive linking / rpath in conan 2.0.x
        replace_in_file(self, os.path.join(self.source_folder, "cmake", "pcl_find_boost.cmake"),
            "set(BOOST_REQUIRED_MODULES filesystem iostreams system)",
            "set(BOOST_REQUIRED_MODULES headers filesystem iostreams system)")
        replace_in_file(self, os.path.join(self.source_folder, "cmake", "pcl_find_boost.cmake"),
            "find_package(Boost 1.65.0 QUIET COMPONENTS serialization mpi)",
            "find_package(Boost 1.81.0 QUIET CONFIG COMPONENTS serialization mpi)")
        replace_in_file(self, os.path.join(self.source_folder, "cmake", "pcl_find_boost.cmake"),
            "find_package(Boost 1.65.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})",
            """find_package(Boost 1.81.0 REQUIRED CONFIG COMPONENTS ${BOOST_REQUIRED_MODULES})\nmessage(STATUS "Boost Include: ${Boost_INCLUDE_DIR}")\ninclude_directories(${Boost_INCLUDE_DIR})""")


        replace_in_file(self, os.path.join(self.source_folder, "CMakeLists.txt"),
            "find_package(FLANN 1.9.1 REQUIRED)",
            """find_package(FLANN 1.9.1 REQUIRED CONFIG)""")

        if self.options.force_cuda_arch:
            replace_in_file(self, os.path.join(self.source_folder, "cmake", "pcl_find_cuda.cmake"),
                """set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")""",
                """
                option(PCL_FORCE_CUDA_ARCH "Option to force CUDA Architectures to be built." "")
                if (PCL_FORCE_CUDA_ARCH)
                  set(__CUDA_ARCH_BIN ${PCL_FORCE_CUDA_ARCH})
                endif (PCL_FORCE_CUDA_ARCH)
                set(CUDA_ARCH_BIN ${__CUDA_ARCH_BIN} CACHE STRING "Specify 'real' GPU architectures to build binaries for")
                """)

        if self.settings.os == "Windows":
            replace_in_file(self, os.path.join(self.source_folder, "CMakeLists.txt"),
                """if("${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""",
                """if("${CONAN_SETTINGS_OS}" STREQUAL "Windows" OR "${CMAKE_CXX_FLAGS}" STREQUAL "${CMAKE_CXX_FLAGS_DEFAULT}")""")







        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()


    @property
    def _pcl_components(self):
        def eigen():
            return ["eigen::eigen"]

        def flann():
            return ["flann::flann"] # + ["lz4:lz4"]

        def boost():
            return ["boost::boost"]

        def cuda_sdk():
            return ["cuda_dev_config::cuda_dev_config"] if self.options.with_cuda else []

        pcl_components = [
            {"target": "pcl_common",          "lib": "common",          "requires": boost() + eigen()},
            {"target": "pcl_io_ply",          "lib": "io_ply",          "requires": ["pcl_common"] + boost() + eigen()},
            {"target": "pcl_io",              "lib": "io",              "requires": ["pcl_common", "pcl_io_ply"] + boost() + eigen()},
            {"target": "pcl_ml",              "lib": "ml",              "requires": ["pcl_common"] + eigen()},
            {"target": "pcl_octree",          "lib": "octree",          "requires": ["pcl_common", "pcl_gpu_containers", "pcl_gpu_utils"] + eigen()},
            {"target": "pcl_kdtree",          "lib": "kdtree",          "requires": ["pcl_common"] + flann() + eigen()},
            {"target": "pcl_search",          "lib": "search",          "requires": ["pcl_common", "pcl_kdtree", "pcl_octree"] + eigen()},
            {"target": "pcl_sample_consensus","lib": "sample_consensus","requires": ["pcl_common", "pcl_search"] + eigen()},
            {"target": "pcl_stereo",          "lib": "stereo",          "requires": ["pcl_common", "pcl_io"] + eigen()},
            {"target": "pcl_surface",         "lib": "surface",         "requires": ["pcl_common", "pcl_search", "pcl_kdtree", "pcl_octree"] + eigen()},
            {"target": "pcl_filters",         "lib": "filters",         "requires": ["pcl_common", "pcl_sample_consensus", "pcl_search", "pcl_kdtree", "pcl_octree"] + eigen()},
            {"target": "pcl_2d",              "lib": None,              "requires": ["pcl_common", "pcl_filters"] + eigen()},
            {"target": "pcl_geometry",        "lib": None,              "requires": ["pcl_common"] + eigen()},
            {"target": "pcl_features",        "lib": "features",        "requires": ["pcl_common", "pcl_search", "pcl_kdtree", "pcl_octree", "pcl_filters", "pcl_2d"] + eigen()},
            {"target": "pcl_segmentation",    "lib": "segmentation",    "requires": ["pcl_common", "pcl_geometry", "pcl_search", "pcl_sample_consensus", "pcl_kdtree", "pcl_octree", "pcl_features", "pcl_filters", "pcl_ml"] + eigen()},
            {"target": "pcl_tracking",        "lib": "tracking",        "requires": ["pcl_common", "pcl_octree", "pcl_kdtree", "pcl_search", "pcl_sample_consensus", "pcl_features", "pcl_filters"] + eigen()},
            {"target": "pcl_registration",    "lib": "registration",    "requires": ["pcl_common", "pcl_search", "pcl_kdtree", "pcl_octree", "pcl_filters", "pcl_2d"] + eigen()},
            {"target": "pcl_keypoints",       "lib": "keypoints",       "requires": ["pcl_common", "pcl_search", "pcl_kdtree", "pcl_octree", "pcl_features", "pcl_filters"] + eigen()},
            {"target": "pcl_recognition",     "lib": "recognition",     "requires": ["pcl_common", "pcl_io", "pcl_search", "pcl_kdtree", "pcl_octree", "pcl_features", "pcl_filters", "pcl_registration", "pcl_sample_consensus", "pcl_ml"] + eigen()},
        ]

        if self.options.with_cuda:
            pcl_components.extend([
            {"target": "pcl_gpu_containers",    "lib": "gpu_containers",    "requires": cuda_sdk()},
            {"target": "pcl_gpu_utils",         "lib": "gpu_utils",         "requires": ["pcl_gpu_containers"]},
            {"target": "pcl_cuda_features",     "lib": "cuda_features",     "requires": ["pcl_common", "pcl_io", "cuda_dev_config::cuda_dev_config"] + boost()},
            {"target": "pcl_cuda_segmentation", "lib": "cuda_segmentation", "requires": ["cuda_dev_config::cuda_dev_config"] + boost()},
            ])


        return pcl_components



    # def package_info(self):
    #     self.cpp_info.libs = collect_libs(self)
    #     v_major, v_minor, v_micro = self.upstream_version.split(".")
    #     self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]


    @property
    def _module_file_rel_path(self):
        return os.path.join("lib", "cmake", f"conan-official-{self.name}-targets.cmake")


    def package_info(self):
        version = self.version.split(".")
        # version = "".join(version) if self.settings.os == "Windows" else ""
        # debug = "d" if self.settings.build_type == "Debug" and self.settings.os == "Windows" else ""
        debug = ""

        def get_lib_name(module):
            return f"pcl_{module}{debug}"

        def add_components(components):
            for component in components:
                conan_component = component["target"]
                cmake_target = component["target"]
                cmake_component = component["lib"]
                lib_name = None
                if cmake_component is None:
                    # header only library
                    lib_name = None
                else:

                    lib_name = get_lib_name(component["lib"])
                requires = component["requires"]
                # TODO: we should also define COMPONENTS names of each target for find_package() but not possible yet in CMakeDeps
                #       see https://github.com/conan-io/conan/issues/10258
                self.cpp_info.components[conan_component].set_property("cmake_target_name", cmake_target)
                if lib_name is not None:
                    self.cpp_info.components[conan_component].libs = [lib_name]
                # if self.settings.os != "Windows":
                self.cpp_info.components[conan_component].includedirs.append(os.path.join("include", "pcl-{}.{}".format(version[0], version[1])))
                self.cpp_info.components[conan_component].requires = requires
                # if self.settings.os == "Linux":
                #     self.cpp_info.components[conan_component].system_libs = ["dl", "m", "pthread", "rt"]

                # if conan_component == "opencv_core" and not self.options.shared:
                #     lib_exclude_filter = "(opencv_|ippi|correspondence|multiview|numeric).*"
                #     libs = list(filter(lambda x: not re.match(lib_exclude_filter, x), collect_libs(self)))
                #     self.cpp_info.components[conan_component].libs += libs

                # # TODO: to remove in conan v2 once cmake_find_package* generators removed
                # self.cpp_info.components[conan_component].names["cmake_find_package"] = cmake_target
                # self.cpp_info.components[conan_component].names["cmake_find_package_multi"] = cmake_target
                # self.cpp_info.components[conan_component].build_modules["cmake_find_package"] = [self._module_file_rel_path]
                # self.cpp_info.components[conan_component].build_modules["cmake_find_package_multi"] = [self._module_file_rel_path]
                # if cmake_component != cmake_target:
                #     conan_component_alias = conan_component + "_alias"
                #     self.cpp_info.components[conan_component_alias].names["cmake_find_package"] = cmake_component
                #     self.cpp_info.components[conan_component_alias].names["cmake_find_package_multi"] = cmake_component
                #     self.cpp_info.components[conan_component_alias].requires = [conan_component]
                #     self.cpp_info.components[conan_component_alias].bindirs = []
                #     self.cpp_info.components[conan_component_alias].includedirs = []
                #     self.cpp_info.components[conan_component_alias].libdirs = []

        self.cpp_info.set_property("cmake_file_name", "PCL")

        add_components(self._pcl_components)

        self.cpp_info.includedirs.append(os.path.join("include", "pcl-{}.{}".format(version[0], version[1])))
