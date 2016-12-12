# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='floq',
    version='0.0.1',
    description='Floquet control code',
    long_description=readme,
    author='Marcel Langer',
    author_email='me@sirmarcel.com',
    url='https://github.com/sirmarcel/floq',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'benchmark'))
)
