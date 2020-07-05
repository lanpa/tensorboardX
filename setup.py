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
        import os
        os.system("pip install protobuf numpy six")
        install.run(self)

with open('HISTORY.rst') as history_file:
    history = history_file.read()

preparing_PyPI_package = 'sdist' in sys.argv or 'bdist_wheel' in sys.argv
version_git = version = '2.1'

if not preparing_PyPI_package:
    if os.path.exists('.git'):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        version_git = version_git + '+' + sha[:7]

with open('tensorboardX/__init__.py', 'a') as f:
    f.write('\n__version__ = "{}"\n'.format(version_git))

requirements = [
    'numpy',
    'protobuf >= 3.8.0',
    'six',
]

test_requirements = [
    'pytest',
    'matplotlib',
    'crc32c',
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
# update the version number in this file.
# python setup.py sdist bdist_wheel --universal
# check the generated tar.gz file (the version, no *.pyc)
# git add [files]
# git commit -m 'prepare for release'
# add tag
# twine upload dist/*
# push commit