#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
import numpy as np      

#test simple plane data 3 points, 1 wrong point:
X = np.array ([[1., 1.], [2., 1.], [6., 1.]])
Y = np.array ([0, 1, 0])
W = np.array ([1., 0.])
YNEGPOS = bsSvmCheckData (X, Y, W, bsSvmLinKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 4.0 or MARGB12[1] != -2.0 or MARGB12[2] != -6.0 or i1 != 1 or i2 != 2 or cntWrng != 1:
  print ('No met W{1,0}: margin min=4, bHi=-2, bHj=-6, i=1, j=2, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 OK!')

bsSvmInverseY (Y, YNEGPOS)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 4.0 or MARGB12[1] != -2.0 or MARGB12[2] != -6.0 or i1 != 1 or i2 != 2 or cntWrng != 1:
  print ('No met W{1,0}: margin min=4, bHi=-2, bHj=-6, i=1, j=2, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 inv OK!')

bsSvmInverseY (Y, YNEGPOS)
X[1][0] = 4.0
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 3. or MARGB12[1] != -1.0 or MARGB12[2] != -4.0 or i1 != 0 or i2 != 1 or cntWrng != 1:
  print ('No met W{1,0}: margin min=3., bHi=-1, bHj=-4, i=0, j=1, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 2 OK!')

bsSvmInverseY (Y, YNEGPOS)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 3. or MARGB12[1] != -1.0 or MARGB12[2] != -4.0 or i1 != 0 or i2 != 1 or cntWrng != 1:
  print ('No met W{1,0}: margin min=3., bHi=-1, bHj=-4, i=0, j=1, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 2 inv OK!')
