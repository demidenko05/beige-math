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

#test data form MIT MIT6_034F10_tutor05.pdf page 8:

#K(X,W)=2*mag(X)*mag(W)
def kernelMit (pVEC1, pVEC2):
  magV1 = 0.
  for i in range (pVEC1.shape[0]):
    magV1 += pVEC1[i]**2
  magV1 = np.sqrt(magV1)
  magV2 = 0.
  for i in range (pVEC2.shape[0]):
    magV2 += pVEC2[i]**2
  magV2 = np.sqrt(magV2)
  rz = 2. * magV1 * magV2
  return rz

X = np.array ([[.0, .0], [4., 4.], [-1., -1.], [2., 2.], [-3., -3.], [1., 1.], [-4., -4.]])
Y = np.array ([1, 0, 1, 1, 0, 1, 0])
W = np.array ([1., 0.])
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, W, kernelMit)
MARGB12m = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
W[1] = 1.
MARGB12 = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
W[0] = 0.
MARGB12 = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = int(MARGB12m[3]) + 1
j = int(MARGB12m[4]) + 1
err = 0
print ('T1 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T1 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], MARGB12m[3], MARGB12m[4])
print ('T1 b= ', b)
if not( ( i == 4 and j == 5) or ( i == 5 and j == 4) ):
  err = 1
  print ("i, j != 4, 5! - ", i , j)
#b = -7.0710678118654755, check classification
negposXm2m2 = kernelMit (np.array ([-2., -2.]), Wm) + b
negposXm2m3 = kernelMit (np.array ([-2., -3.]), Wm) + b
print ('T1 classification X(-2,-2) and X(-2, -3): ', negposXm2m2, negposXm2m3)
if not( ( negposXm2m2 < 0. and negposXm2m3 > 0.) or ( negposXm2m2 > 0. and negposXm2m3 < 0.) ):
  err = 1

if err == 0:
  print ('Test 1 OK!')

#T2 inverse Y:
bsSvmInverseY (Y, YNEGPOS)
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, W, kernelMit)
MARGB12m = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
W[1] = 1.
MARGB12 = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
W[0] = 0.
MARGB12 = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = int(MARGB12m[3]) + 1
j = int(MARGB12m[4]) + 1
err = 0
print ('T2 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T2 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], MARGB12m[3], MARGB12m[4])
print ('T2 b= ', b)
if not( ( i == 4 and j == 5) or ( i == 5 and j == 4) ):
  err = 1
  print ("i, j != 4, 5! - ", i , j)
#b = -7.0710678118654755, check classification
negposXm2m2 = kernelMit (np.array ([-2., -2.]), Wm) + b
negposXm2m3 = kernelMit (np.array ([-2., -3.]), Wm) + b
print ('T2 classification X(-2,-2) and X(-2, -3): ', negposXm2m2, negposXm2m3)
if not( ( negposXm2m2 < 0. and negposXm2m3 > 0.) or ( negposXm2m2 > 0. and negposXm2m3 < 0.) ):
  err = 1

if err == 0:
  print ('Test 2 OK!')
