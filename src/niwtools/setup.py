from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
   name='niwtools',
   version='1.0',
   description='A module containing tools to analyse EM-APEX floats',
   author='Sebastian Essink',
   author_email='sebastianessink@gmail.com',
   packages=['niwtools'],  #same as name
   # install_requires=['pandas', 'numpy'], #external packages as dependencies,
   zip_safe=False
)
