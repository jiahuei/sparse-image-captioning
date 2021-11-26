import os
import pkg_resources
from setuptools import setup

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(path):
    with open(path, "r") as f:
        return [_ for _ in f.readlines() if _[:1].isidentifier()]


if __name__ == "__main__":
    package = "sparse_caption"
    requirements = read_requirements_file(os.path.join(CURR_DIR, "requirements_base.txt"))
    requirements += read_requirements_file(os.path.join(CURR_DIR, "requirements.txt"))
    requirements_dev = read_requirements_file(os.path.join(CURR_DIR, "requirements_dev.txt"))
    requirements_extra = list(set(requirements_dev) - set(requirements))
    with open(os.path.join(CURR_DIR, package, "version.py"), "r") as f:
        version = f.readlines()[-1].split('"')[1]

    setup(
        name=package,
        version=version,
        description="",
        author="CiSIPLab Universiti Malaya, Jia-Huei Tan",
        packages=[package],
        python_requires=">=3.7",
        install_requires=[str(r) for r in pkg_resources.parse_requirements(requirements)],
        include_package_data=True,
        extras_require={"dev": requirements_extra},
    )
