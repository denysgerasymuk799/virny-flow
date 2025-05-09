import os
import pathlib
import pkg_resources

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

NAME = 'virny_flow'
DESCRIPTION = "Design space for responsible model development in Python"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/denysgerasymuk799/virny-flow/"
EMAILS = "denis.gerasymuk799@gmail.com"
AUTHORS = "Denys Herasymuk"

# The directory containing this file
HERE = os.getcwd()

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

about: dict = {}
with open(os.path.join(HERE, NAME, "__version__.py")) as f:
    exec(f.read(), about)

with pathlib.Path('requirements.txt').open() as lib_base_packages_txt:
    base_packages = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(lib_base_packages_txt)
    ]

# This call to setup() does all the work
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    author=AUTHORS,
    author_email=EMAILS,
    license="BSD-3",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=base_packages,
    extras_require={
        "dev": base_packages
               + [
                   "pytest~=7.2.2",
                   "python-dotenv~=1.0.0",
                   "lightgbm==4.3.0",
                   "folktables~=0.0.11",
               ],
        "docs": [
            "scikit-learn",
            "numpy",
            "scipy",
            "pandas",
            "dominate",
            "flask",
            "ipykernel",
            "jupyter-client",
            "mike",
            "mkdocs",
            "mkdocs-awesome-pages-plugin",
            "mkdocs-material",
            "mkdocs-redirects",
            "nbconvert",
            "python-slugify",
            "spacy",
        ],
    },
    python_requires='>=3.9',
)
