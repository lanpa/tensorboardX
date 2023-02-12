#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

# Dynamically compile protos
def compileProtoBuf():
    res = subprocess.call(['bash', './compile.sh'])
    assert res == 0, 'cannot compile protobuf'

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        compileProtoBuf()
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        compileProtoBuf()
        for r in requirements:
            subprocess.run(f"pip install '{r}'", shell=True)
        install.run(self)

with open('HISTORY.rst') as history_file:
    history = history_file.read()

preparing_PyPI_package = 'sdist' in sys.argv or 'bdist_wheel' in sys.argv
version_git = version = subprocess.check_output(['git', 'describe', '--always']).decode('ascii').strip()

# pass version without using argparse
# format example: v1.2.3
publish_version = sys.argv[-1]
if publish_version[0] == 'v':
    version_git = publish_version[1:]
    sys.argv = sys.argv[:-1]
print(version_git)

if not preparing_PyPI_package:
    if os.path.exists('.git'):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        version_git = version_git + '+' + sha[:7]

with open('tensorboardX/__init__.py', 'a') as f:
    f.write('\n__version__ = "{}"\n'.format(version_git))

requirements = [
    'numpy',
    'packaging',
    'protobuf>=3.8.0,<4',
]


setup(
    name='tensorboardX',
    version=version_git,
    description='TensorBoardX lets you watch Tensors Flow without Tensorflow',
    long_description=history,
    author='Tzu-Wei Huang',
    author_email='huang.dexter@gmail.com',
    url='https://github.com/lanpa/tensorboardX',
    packages=['tensorboardX'],
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)


# checklist: update History.rst readme.md
# update the version number in this file (setup.py).
# python setup.py sdist bdist_wheel --universal
# check the generated tar.gz file 
# (1. The version number is correct. 2. no *.pyc __pycache__ files)
# git checkout -b "release x.x"
# git add setup.py History.rst readme.md (skip tensorboardX/__init__.py)
# git commit -m 'prepare for release'
# add tag
# twine upload dist/*
# git push -u origin HEAD
