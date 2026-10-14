from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(path: str) -> List[str]:
    with open(path) as f:
        reqs = [r.strip() for r in f.readlines() if r.strip()]
    return [r for r in reqs if r != HYPHEN_E_DOT]

setup(
    name="industrial-defect-detection",
    version="0.1.0",
    author="Your Name",
    description="PatchCore anomaly detection on MVTec AD metal_nut",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)