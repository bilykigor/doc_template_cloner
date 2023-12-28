
from __future__ import absolute_import

import os
from glob import glob

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_requirements(file_content):
    lines = file_content.splitlines()
    return [line.strip() for line in lines if line and not line.startswith("#")]


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open(os.path.join(HERE, "requirements.txt")) as f:
    requirements = parse_requirements(f.read())
    

setup(
    name="doc_template_cloner",
    version="0.2.2",
    description="Custom lib to clone lebaling from one image to another similar.",
    long_description_content_type="text/markdown",
    long_description=read('README.md'),
    packages=find_packages(),
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("doc_template_cloner/*.py")],
    include_package_data=True,
    author="Ihor Bilyk",
    url="https://github.com/bilykigor/doc_template_cloner",
    license="Apache License 2.0",
    keywords="Custom lib to clone lebaling from one image to another similar",
    python_requires=">= 3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements
)
