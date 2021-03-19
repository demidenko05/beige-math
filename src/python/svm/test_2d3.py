#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
from BsLibMisc import *
import numpy as np      

bsSvmLinKern = BsSvmLinKern ()

#test simple plane data 6 points, 1 wrong point:
X = np.array ([[1., 1.], [2., 1.], [3., 1.], [13., 1.], [14., 1.], [15., 1.]])
Y = np.array ([0, 1, 0, 1, 1, 1])
W = np.array ([1., 0.])
YNEGPOS = bsSvmCheckData (X, Y, bsSvmLinKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 OK!')

bsSvmInverseY (Y, YNEGPOS)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 inv OK!')

bsSvmInverseY (Y, YNEGPOS)
Y[1] = 0
Y[4] = 0
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 2 OK!')

bsSvmInverseY (Y, YNEGPOS)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 2 inv OK!')
