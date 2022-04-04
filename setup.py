#!/usr/bin/env python
"""Setup file for the gradient_metrics package."""


from gradient_metrics import __version__
from setuptools import setup

with open("README.md") as f:
    readme = f.read()


def parse_requirements(requirements, ignore=("setuptools",)):
    """Read dependencies from requirements file (with version numbers if any)

    Note: this implementation does not support requirements files with extra
    requirements
    """
    with open(requirements) as f:
        packages = set()
        for line in f:
            line = line.strip()
            if line.startswith(("#", "-r", "--")):
                continue
            if "#egg=" in line:
                line = line.split("#egg=")[1]
            pkg = line.strip()
            if pkg not in ignore:
                packages.add(pkg)
        return tuple(packages)


setup(
    name="gradient_metrics",
    version=__version__,
    description="",
    author="Philipp Oberdiek",
    author_email="git@oberdiek.net",
    long_description=readme,
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.6",
    tests_require=parse_requirements("requirements-dev.txt"),
    keywords="pytorch machine-learning deep-learning uncertainty",
)
