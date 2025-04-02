from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sourceid-nmf",
    version="0.1.0",
    author="Original Authors",
    author_email="zhuang82-c@my.cityu.edu.hk",
    description="Microbial source tracking via non-negative matrix factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/sourceid-nmf",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "tqdm>=4.66.1",
        "scipy>=1.10.1",
    ],
    entry_points={
        "console_scripts": [
            "sourceid-nmf=sourceid_nmf.cli:main",
        ],
    },
)