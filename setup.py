import os
import pkg_resources
from setuptools import setup


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as f:
        requirements = [_ for _ in f.readlines() if _[:1].isidentifier()]
    with open(os.path.join(os.path.dirname(__file__), "captioning", "version.py"), "r") as f:
        version = f.readlines()[-1].split('"')[1]

    setup(
        name="captioning",
        version=version,
        description="",
        author="Jia-Huei",
        packages=["captioning"],
        python_requires=">=3.6",
        install_requires=[str(r) for r in pkg_resources.parse_requirements(requirements)],
        include_package_data=True,
        extras_require={"dev": ["pytest"]},
    )
