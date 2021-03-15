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
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
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
if not( ( i == 4 and j == 5) or ( i == 5 and j == 4) ):
  print ("i, j != 4, 5! - ", i , j)
  exit (1)
#b = -7.0710678118654755, check classification
negposXm2m2 = kernelMit (np.array ([-2., -2.]), Wm) + b
negposXm2m3 = kernelMit (np.array ([-2., -3.]), Wm) + b
print ('T1 classification X(-2,-2) and X(-2, -3): ', negposXm2m2, negposXm2m3)
if not( ( negposXm2m2 < 0. and negposXm2m3 > 0.) or ( negposXm2m2 > 0. and negposXm2m3 < 0.) ):
  print ('No met: negposXm2m2<0, negposXm2m3>0', negposXm2m2,  negposXm2m3)

print ('Test 1 OK!')

#T2 inverse Y:
bsSvmInverseY (Y, YNEGPOS)
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, W, kernelMit)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = i1m + 1
j = i2m + 1
err = 0
print ('T2 Wmax w1, w2: ', Wm[0], Wm[1])
print ('T2 margin max, bHi, bHj, i, j: ', MARGB12m[0], MARGB12m[1], MARGB12m[2], i1m, i2m)
print ('T2 b= ', b)
if not( ( i == 4 and j == 5) or ( i == 5 and j == 4) ):
  print ("i, j != 4, 5! - ", i , j)
  exit (1)
#b = -7.0710678118654755, check classification
negposXm2m2 = kernelMit (np.array ([-2., -2.]), Wm) + b
negposXm2m3 = kernelMit (np.array ([-2., -3.]), Wm) + b
print ('T2 classification X(-2,-2) and X(-2, -3): ', negposXm2m2, negposXm2m3)
if not( ( negposXm2m2 < 0. and negposXm2m3 > 0.) or ( negposXm2m2 > 0. and negposXm2m3 < 0.) ):
  print ('No met: negposXm2m2<0, negposXm2m3>0', negposXm2m2,  negposXm2m3)
  exit (1)

print ('Test 2 OK!')

bsSvmInverseY (Y, YNEGPOS)

W[0] = 1.0
W[1] = -0.5
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
print ('W{1,-0.5} MARGB12: ', MARGB12) #[ 2.82842712 -6.32455532 -9.48683298]
W[0] = -0.5
W[1] = 1.0
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, kernelMit, YNEGPOS)
if MARGB12[0] > MARGB12m[0]:
  print ('No met: W{-0.5,1}MARGB12[0] > W{1,0}MARGB12m[0]', MARGB12[0], MARGB12m[0])
  #exit (1)

b = MARGB12[2] - ((MARGB12[2] - MARGB12[1]) / 2.)

negposXm2m2 = kernelMit (np.array ([-2., -2.]), W) + b
negposXm2m3 = kernelMit (np.array ([-2., -3.]), W) + b
print ('T2 classification X(-2,-2) and X(-2, -3): ', negposXm2m2, negposXm2m3)
if not( ( negposXm2m2 < 0. and negposXm2m3 > 0.) or ( negposXm2m2 > 0. and negposXm2m3 < 0.) ):
  print ('No met: negposXm2m2<0, negposXm2m3>0', negposXm2m2,  negposXm2m3)
  exit (1)

print ('Test 3 OK!')

Wa, ba, cntWrnga = bsSvmTrain (X, Y, kernelMit, YNEGPOS, 0.0001)
#No met: W{-0.5,1}MARGB12[0] > W{1,0}MARGB12m[0] 2.82842712474619 2.828427124746189
#MARGB12m, cntItr, cntWrngMin, cntWrngMax:  [ 2.82842712 -6.32455532 -9.48683298] 28 0 0
#No met: W{-0.5,1}, b=-10, cntWrng=0 [-0.5  1. ] -7.905694150420949 0

if Wa[0] != -0.5 or Wa[1] != 1.0 or ba != -7.905694150420949 or cntWrnga != 0:
  print ("No met: W{-0.5,1}, b=-10, cntWrng=0", Wa, ba, cntWrnga)
  exit (1)

print ('Test 4 OK!')

#!!!!!! here are several decisions by digits, but hyperplane the same, margins: 2.82842712474619 vs 2.828427124746189:
# X{1,0} - 7.0710678118654755 = 0
# X{-0.5,1} -7.905694150420949 = 0
# X{1,-0.5} - 7.905694155 = 0
