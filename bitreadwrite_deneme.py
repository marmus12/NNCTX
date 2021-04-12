#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:56:15 2021

@author: root
"""
from array import array

bin_array = array("B")
bits = "101111111111111110111"

bits = bits + "0" * (32 - len(bits))  # Align bits to 32, i.e. add "0" to tail
for index in range(0, 32, 8):
    byte = bits[index:index + 8][::-1]
    bin_array.append(int(byte, 2))

with open("test.bnr", "wb") as f:
    f.write(bytes(bin_array))