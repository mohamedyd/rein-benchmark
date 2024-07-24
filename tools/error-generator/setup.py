
from setuptools import setup, find_packages

setup(name='errorgenerator',
      version='0.2',
      description='Package to generate highly realistic errors',
      license="TU-Berlin",
      author='Milad Abbaszadeh',
      author_email='milad.abbaszadeh94@gmail.com',
      packages=find_packages(),
      url="https://github.com/BigDaMa/error-generator",
      install_requires=['pandas','numpy','chainer'] 
      )
