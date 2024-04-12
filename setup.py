
from setuptools import setup, find_packages

setup(
    name='roboloader',
    version='0.1',
    author='laureote loic',
    author_email='laureote-loic@hotmail.fr',
    description='dataloader for roboflow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hackolite/roboloader.git',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'pycocotools'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)