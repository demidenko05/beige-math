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

#test data form the first article calc.py:
X = np.array ([[.5, 1.2], [1., 1.], [3., 3.], [4., 3.3]])
Y = np.array ([0, 0, 1, 1])
W = np.array ([1., 0.])
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, bsSvmLinKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 2.0 or MARGB12[1] != -1.0 or MARGB12[2] != -3.0 or i1 != 1 or i2 != 2:
  print ('No met W{1,0}: margin min=2, bHi=-1, bHj=-3, i=1, j=2: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2)
  exit(1)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = i1m + 1
j = i2m + 1
print ('T1 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T1 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], i1m, i2m)
print ('T1 b= ', b)
if b != -4.:
  print ("b != -4! b = %s " % b)
  exit(1)

if not( ( i == 2 and j == 3) or ( i == 3 and j == 2) ):
  print ("i, j != 2, 3! - ", i , j)
  exit(1)

print ('Test1 OK!')

#T2 inverse Y:
bsSvmInverseY (Y, YNEGPOS);
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 2.0 or MARGB12[1] != -1.0 or MARGB12[2] != -3.0 or i1 != 1 or i2 != 2:
  print ('No met W{1,0}: margin min=2, bHi=-1, bHj=-3, i=1, j=2: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2)
  exit(1)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = i1m + 1
j = i2m + 1
print ('T2 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T2 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], i1m, i2m)
print ('T2 b= ', b)
if b != -4.:
  print ("b != -4! b = %s " % b)
  exit(1)

if not( ( i == 2 and j == 3) or ( i == 3 and j == 2)  ):
  print ("i, j != 2, 3! - ", i , j)
  exit(1)

print ('Test2 OK!')

#T3 broke the last sample:
bsSvmInverseY (Y, YNEGPOS);
lstIdx = Y.shape[0] - 1
for i in range (Y.shape[0]):
  if Y[lstIdx] != YNEGPOS[i]:
    Y[lstIdx] = YNEGPOS[i]
    break
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 2.0 or MARGB12[1] != -1.0 or MARGB12[2] != -3.0 or i1 != 1 or i2 != 2 or cntWrng != 1:
  print ('No met W{1,0}: margin min=2, bHi=-1, bHj=-3, i=1, j=2, cntWrong=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = i1m + 1
j = i2m + 1
print ('T3 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T3 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], i1m, i2m)
print ('T3 b= ', b)
if b != -4.:
  print ("b != -4! b = %s " % b)
  exit(1)

if not( ( i == 2 and j == 3) or ( i == 3 and j == 2)  ):
  print ("i, j != 2, 3! - ", i , j)
  exit(1)

print ('Test3 OK!')

#T4 broke the first sample:
for i in range (Y.shape[0]):
  if Y[lstIdx] != YNEGPOS[i]:
    Y[lstIdx] = YNEGPOS[i]
    break
bsSvmInverseY (Y, YNEGPOS);
for i in range (Y.shape[0]):
  if Y[0] != YNEGPOS[i]:
    Y[0] = YNEGPOS[i]
    break
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12[0] != 2.0 or MARGB12[1] != -1.0 or MARGB12[2] != -3.0 or i1 != 1 or i2 != 2 or cntWrng != 1:
  print ('No met W{1,0}: margin min=2, bHi=-1, bHj=-3, i=1, j=2, cntWrong=1: ', MARGB12[0], MARGB12[1], MARGB12[2], i1, i2, cntWrng)
  exit(1)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = i1m + 1
j = i2m + 1
print ('T4 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T4 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], i1m, i2m)
print ('T4 b= ', b)
if b != -4.:
  print ("b != -4! b = %s " % b)
  exit(1)

if not( ( i == 2 and j == 3) or ( i == 3 and j == 2)  ):
  print ("i, j != 2, 3! - ", i , j)
  exit(1)

print ('Test4 OK!')

for i in range (Y.shape[0]):
  if Y[0] != YNEGPOS[i]:
    Y[0] = YNEGPOS[i]
    break

Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, bsSvmLinKern, YNEGPOS, 0.0001)

if Wa[0] != 1.0 or Wa[1] != 1.0 or ba != -4.0 or cntWrnga != 0:
  print ("No met: W{1,1}, b=-4, cntWrng=0", Wa, ba, cntWrnga)
  exit(1)

print ('Test 5 OK!')

