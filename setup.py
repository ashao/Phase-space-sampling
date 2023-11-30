import os

from setuptools import setup

setup(
    name="uips",
    version="0.0.7",
    description="Reduce a large and high-dimensional dataset by downselecting data uniformly in phase space",
    url="https://github.com/NREL/Phase-space-sampling",
    author="Malik Hassanaly",
    license="BSD 3-Clause",
    package_dir={"uips": "uips"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: BSD 3 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={"": ["input2D", "input2D_bins"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "openmpi",
        "torch",
        "numpy",
        "mpi4py",
        "scikit-learn",
        "prettyPlot>=0.0.10",
        "black",
        "isort",
        "codespell",
    ],
)