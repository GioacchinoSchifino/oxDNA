"""Legacy setuptools entry point; modern builds use pyproject.toml/scikit-build-core."""
import os
from setuptools import setup

setup(
    install_requires = [
        f"oxDNA_analysis_tools @ file://localhost/{os.getcwd()}/analysis/"
    ],
)
