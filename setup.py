#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

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
        import os
        os.system("pip install protobuf numpy six")
        install.run(self)

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# comment if preparing PyPI package
version = '1.7'
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
version_git = version + '+' + sha[:7]

with open('tensorboardX/__init__.py', 'a') as f:
    f.write('\n__version__ = "{}"\n'.format(version_git))
# end of comment

requirements = [
    'numpy',
    'protobuf >= 3.2.0',
    'six',
]

test_requirements = [
    'pytest',
    'matplotlib'
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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    test_suite='tests',
    tests_require=test_requirements
)


# checklist: update History.rst readme.md
# version=version_git <--- change to sha-less version (in setup.py)
# __version__ = "1.x" (__init__.py)
# commit
# add tag
# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
# push commit