from setuptools import setup, find_packages
import os


# Utility function to read the README file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read the Requirements file
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

setup(
    name='rein',
    version='0.1.0',
    author='Mohamed Abdelaal, Christian Hammacher',
    author_email='mohamed.abdelaal@softwareag.com, christian.hammacher@softwareag.com',
    description='Benchmarking data cleaning methods through evaluating their impact on a set of ML models',
    long_description=read('README.md'),
    packages=['rein', 'rein.auxiliaries', 'tools', 'tools.Profiler'],
    #install_requires=requirements,
    classifiers=['Development Status :: 4 - Beta'],
)
