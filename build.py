from conan.packager import ConanMultiPackager
import os

if __name__ == "__main__":
    builder = ConanMultiPackager(
        username="sight", 
        visual_runtimes=["MD", "MDd"],
        archs=["x86_64"])
    if os.getenv("CONAN_USE_CUDA", False):
        builder.add({"arch": "x86_64", "build_type": "Debug"},
                    {"pcl:shared": True, "pcl:use_cuda": True})
        builder.add({"arch": "x86_64", "build_type": "Release"},
                    {"pcl:shared": True, "pcl:use_cuda": True})
    else:
        builder.add({"arch": "x86_64", "build_type": "Debug"},
                    {"pcl:shared": True, "pcl:use_cuda": False})
        builder.add({"arch": "x86_64", "build_type": "Release"},
                    {"pcl:shared": True, "pcl:use_cuda": False})
    builder.run()