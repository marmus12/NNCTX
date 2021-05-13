#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:12:44 2021

@author: root
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("cython_dene.pyx"),
    zip_safe=False,
)
