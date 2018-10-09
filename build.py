from conan.packager import ConanMultiPackager
import os

if __name__ == "__main__":
    builder = ConanMultiPackager(
        username="sight", 
        visual_runtimes=["MD", "MDd"],
        archs=["x86_64"])
    builder.add_common_builds(shared_option_name=False, pure_c=True)
    filtered_builds = []
    use_cuda = (os.getenv("CONAN_USE_CUDA", "False") == "True")
    for settings, options, env_vars, build_requires, reference in builder.items:
        options["pcl:use_cuda"] = use_cuda
        filtered_builds.append([settings, options, env_vars, build_requires])
    builder.builds = filtered_builds
    builder.run()