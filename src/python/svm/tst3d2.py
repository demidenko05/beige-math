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

#test simple plane data 6 points 3d, 1 wrong:
X = np.array ([[1., 1., 1.], [2., 1., 1.], [3., 1., 1.], [13., 1., 1.], [14., 1., 1.], [15., 1., 1.]])
Y = np.array ([0, 1, 0, 1, 1, 1])
W = np.array ([1., 0., 0.])
YNEGPOS = bsSvmCheckData (X, Y, W, bsSvmLinKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 OK!')

bsSvmInverseY (Y, YNEGPOS)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 10.0 or MARGB12[1] != -3.0 or MARGB12[2] != -13.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,0,0}: margin min=10, bHi=-3, bHj=-13, i=2, j=3, conWrn=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)

print ('Tests 1 inv OK!')

bsSvmInverseY (Y, YNEGPOS)
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] >= 10.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,1,0}: margin min<10, i=2, j=3, conWrn=1: ', MARGB12[0], i1, i2, cntWrng)
  exit(1)
print ('W{1,1,0} MARGB12: ', MARGB12)
print ('Tests 2 OK!')

W[2] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] >= 10.0 or i1 != 2 or i2 != 3 or cntWrng != 1:
  print ('No met W{1,1,1}: margin min<10, i=2, j=3, conWrn=1: ', MARGB12[0], i1, i2, cntWrng)
  exit(1)
print ('W{1,1,1} MARGB12: ', MARGB12)
print ('Tests 3 OK!')

Wa, ba, cntWrnga = bsSvmTrain (X, Y, bsSvmLinKern, YNEGPOS, 0.0001)

#MARGB12m, cntItr, cntWrngMin, cntWrngMax:  [ 9.99999993 12.99829102  2.99951172] 42 1 2
if Wa[0] != 1.0 or Wa[1] != 0.0 or Wa[1] != 0.0 or (-8.0 - ba) > 0.0001 or cntWrnga != 1:
  print ("No met: W{1,0,0}, (-8.0 - b) <= 0.0001, cntWrng=1", Wa, ba, cntWrnga)
  exit(1)

print ('Test 4 OK!')
