from setuptools import setup, find_packages

setup(
    name="nirvana-quant",
    version="1.0.0",
    author="yinyusi65",
    description="Nirvana: Chrono-Cancer Meta-Architecture for High-Dimensional Cyclical Time-Series & Dynamical Network Analysis",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scipy>=1.8.0",
    ],
    python_requires=">=3.9",
)
