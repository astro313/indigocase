from setuptools import setup
from glob import glob
from setuptools import find_packages
from os.path import *

setup(name='indigocase',
      version='1.0',
      description='Code base for Indigo case',
      url='http://github.com/astro313/indigocase',
      author='Daisy Leung',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['markdown'],
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      zip_safe=False)
