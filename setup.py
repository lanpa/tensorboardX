#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

# Dynamically compile protos
def compileProtoBuf():
    res = subprocess.call(['bash', './compile.sh'])
    assert res == 0, 'cannot compile protobuf'

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        for r in requirements:
            subprocess.run(f"pip install '{r}'", shell=True)
        install.run(self)

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'packaging',
    'protobuf',
]


setup(
    name='tensorboardX',
    description='TensorBoardX lets you watch Tensors Flow without Tensorflow',
    long_description=history,
    author='Tzu-Wei Huang',
    author_email='huang.dexter@gmail.com',
    url='https://github.com/lanpa/tensorboardX',
    packages=['tensorboardX'],
    include_package_data=True,
    install_requires=requirements,
    use_scm_version={
        'write_to': "tensorboardX/_version.py",
    },
    setup_requires=['setuptools_scm'],
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
        'Programming Language :: Python :: 3.11',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)


# pypi checklist: 
# update History.rst readme.md
# git tag -a v1.0.0 -m "version 1.0.0"
# git push -u origin HEAD --tags
# go github actions, enter version number v1.0.0 then publish

