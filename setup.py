#!/usr/bin/env python

from setuptools import find_namespace_packages, setup

# Dependencies come from environment.yaml and requirements-nodeps.txt.
setup(
    name="leak-curber",
    version="0.0.1",
    description="Leak-CURBER benchmark: splitters, embedders and baseline code",
    author="BiSECt Lab",
    url="https://github.com/bisect-group/Leak-CURBER",
    license="MIT",
    python_requires=">=3.12",
    packages=find_namespace_packages(
        include=["src", "src.*", "hydra_plugins", "hydra_plugins.*"]
    ),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
        ]
    },
)
