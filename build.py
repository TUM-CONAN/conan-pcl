from conan.packager import ConanMultiPackager

if __name__ == "__main__":
    builder = ConanMultiPackager(
        username="fw4spl", 
        visual_runtimes=["MD", "MDd"],
        archs=["x86_64"])
    builder.add_common_builds(shared_option_name=False, pure_c=True)
    builder.run()