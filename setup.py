from pathlib import Path
from setuptools import setup, find_packages
setup(
    name='basenji',
    version='0.1',
    description='Sequential regulatory activity machine learning',
    author='David Kelley',
    author_email='drk@calicolabs.com',
    url='https://github.com/calico/basenji',
    packages=find_packages(exclude=('tests', 'docs')),
)