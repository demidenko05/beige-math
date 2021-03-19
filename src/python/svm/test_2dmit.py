#!/usr/bin/env python
# coding=UTF-8

#test data form MIT MIT6_034F10_tutor05.pdf page 8 (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)
#license the same

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
from BsLibMisc import *
import numpy as np      


class MitKern:
  #K(X,W)=2*mag(X)*mag(W)
  def dot (self, pVEC1, pVEC2):
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

mitKern = MitKern ()

X = np.array ([[.0, .0], [4., 4.], [-1., -1.], [2., 2.], [-3., -3.], [1., 1.], [-4., -4.]])
Y = np.array ([1, 0, 1, 1, 0, 1, 0])
Xt = np.array ([[-2., -2.], [-2., -3.], [2., 2.], [2., 3.], [3., 3.], [-3., -3.], [0., -13.], [0., 13.], [0.5, 1.25]])
Yt = np.array ([1, 0, 1, 0, 0, 0, 0, 0, 1])

W = np.array ([1., 0.])
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, mitKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
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
if not( i == 4 and j == 5 ):
  print ("i, j != 4, 5! - ", i , j)
  exit (1)
#b = -7.0710678118654755, check classification
for i in range (X.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (X[i], Wm) + b) >= 0:
    cls = YNEGPOS[1]
  if Y[i] != cls:
    print ('T1 wrong classification i, Xi, Yi, cls: ', i, X[i], Y[i], cls)
    exit (1)

for i in range (Xt.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (Xt[i], Wm) + b) >= 0:
    cls = YNEGPOS[1]
  if Yt[i] != cls:
    print ('T1 wrong classification i, Xti, Yti, cls: ', i, Xt[i], Yt[i], cls)
    exit (1)

print ('Test 1 OK!')

#T2 inverse Y:
bsSvmInverseY (Y, YNEGPOS)
bsSvmInverseY (Yt, YNEGPOS)
W[0] = 1.
W[1] = 0.
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, mitKern)
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
MARGB12m = MARGB12
i1m = i1
i2m = i2
W[1] = 1.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  i1m = i1
  i2m = i2
  Wm = W.copy ()
W[0] = 0.
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
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
if not( i == 4 and j == 5 ):
  print ("i, j != 4, 5! - ", i , j)
  exit (1)
#b = -7.0710678118654755, check classification

for i in range (X.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (X[i], Wm) + b) >= 0:
    cls = YNEGPOS[1]
  if Y[i] != cls:
    print ('T2 inv wrong classification i, Xi, Yi, cls: ', i, X[i], Y[i], cls)
    exit (1)

for i in range (Xt.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (Xt[i], Wm) + b) >= 0:
    cls = YNEGPOS[1]
  if Yt[i] != cls:
    print ('T2 inv wrong classification i, Xti, Yti, cls: ', i, Xt[i], Yt[i], cls)
    exit (1)

print ('Test 2 inv OK!')

bsSvmInverseY (Y, YNEGPOS)
bsSvmInverseY (Yt, YNEGPOS)

W[0] = 1.0
W[1] = -0.5
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
print ('W{1,-0.5} MARGB12: ', MARGB12) #[ 2.82842712 -6.32455532 -9.48683298]
W[0] = -0.5
W[1] = 1.0
MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (X, Y, W, mitKern, YNEGPOS)
print ('W{-0.5, 1} MARGB12: ', MARGB12) #[ 2.82842712 -6.32455532 -9.48683298]
if MARGB12[0] > MARGB12m[0]:
  print ('No met: W{-0.5,1}MARGB12[0] > W{1,0}MARGB12m[0]', MARGB12[0], MARGB12m[0])
  #exit (1)

b = MARGB12[2] - ((MARGB12[2] - MARGB12[1]) / 2.)

for i in range (X.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (X[i], W) + b) < 0: #inverse against {1,0} -7.08
    cls = YNEGPOS[1]
  if Y[i] != cls:
    print ('T3 wrong classification i, Xi, Yi, cls: ', i, X[i], Y[i], cls)
    exit (1)

for i in range (Xt.shape[0]):
  cls = YNEGPOS[0]
  if (mitKern.dot (Xt[i], W) + b) < 0:
    cls = YNEGPOS[1]
  if Yt[i] != cls:
    print ('T3 wrong classification i, Xti, Yti, cls: ', i, Xt[i], Yt[i], cls)
    exit (1)

print ('Test 3 OK!')

Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, mitKern, YNEGPOS, 0.0001)
#No met: W{-0.5,1}MARGB12[0] > W{1,0}MARGB12m[0] 2.82842712474619 2.828427124746189
#MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [ 2.82842712 -6.32455532 -9.48683298] 28 0 0 3 4
#No met: W{-0.5,1}, b=-10, cntWrng=0 [-0.5  1. ] -7.905694150420949 0

if Wa[0] != -0.5 or Wa[1] != 1.0 or ba != -7.905694150420949 or cntWrnga != 0:
  print ("No met: W{-0.5,1}, b=-10, cntWrng=0", Wa, ba, cntWrnga)
  exit (1)

print ('Test 4 OK!')

#!!!!!! here are several solutions by digits, but classification OK, but classes swapped, margins: 2.82842712474619 vs 2.828427124746189:
# X{1,0} - 7.0710678118654755 = 0 (mitKern.dot (Xt[i], W) + b) >= 0
# X{-0.5,1} -7.905694150420949 = 0 (mitKern.dot (Xt[i], W) + b) < 0
# X{1,-0.5} - 7.905694155 = 0 (mitKern.dot (Xt[i], W) + b) < 0
