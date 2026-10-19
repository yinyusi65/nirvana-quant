from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="nirvana-quant",
    version="1.0.1",
    author="yinyusi65",
    description="A Statistical and Econometric Framework for Multivariate Cyclical Time Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yinyusi65/nirvana-quant",
    packages=find_packages(include=["nirvana", "nirvana.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.22.0,<2.3.0",
        "pandas>=1.4.0,<3.0.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "full": ["scikit-learn>=1.0.0", "matplotlib>=3.5.0", "seaborn>=0.11.0", "networkx>=2.6.0"],
    },
    python_requires=">=3.9",
)
