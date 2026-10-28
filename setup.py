"""Include canonical runtime assets in wheels without duplicate source copies."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "industrial_inspection/_resources"
        for directory, suffixes in (("configs", {".yaml"}), ("web", {".html", ".js", ".css"})):
            for path in sorted((source / directory).rglob("*")):
                if path.is_file() and path.suffix in suffixes:
                    target = destination / path.relative_to(source)
                    self.mkpath(str(target.parent))
                    self.copy_file(str(path), str(target))


setup(cmdclass={"build_py": BuildWithResources})
