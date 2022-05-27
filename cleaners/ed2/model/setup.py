# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='errordetection',
    version='0.0.1',
    description='Semi-supervised Error Detection',
    long_description=readme,
    author='Felix Neutatz',
    author_email='neutatz@gmail.com',
    url='https://github.com/BigDaMa/SequentialPatternErrorDetection',
    license=license,
    package_data={'config': ['ml/configuration/resources']},
    include_package_data=True,
    packages=find_packages(exclude=('tests', 'docs'))
)

