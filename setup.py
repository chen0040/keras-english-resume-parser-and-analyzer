from setuptools import find_packages
from setuptools import setup


setup(name='keras_en_parser_and_analyzer',
      version='0.0.1',
      description='Chinese Parser and Analyzer using recurrent network in Keras',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-english-resume-parser-and-analyzer',
      download_url='https://github.com/chen0040/keras-english-resume-parser-and-analyzer/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras'],
      packages=find_packages())
