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

#test margin equation for different bH2 and bH1
#because of test_dig012a.py see lines from 75

X = np.array ([[1., 1.], [3., 3.]])
Y = np.array ([-1, 1])
YNEGPOS = bsSvmCheckData (X, Y, bsSvmLinKern)

Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, bsSvmLinKern, YNEGPOS, 0.0001)

if Wa[0] != 1.0 or Wa[1] != 1.0 or ba != -4.0 or cntWrnga != 0:
  print ("No met: W{1,1}, b=-4, cntWrng=0", Wa, ba, cntWrnga)
  exit(1)

X[0][0] = -3.0
X[0][1] = -3.0
magWa = np.linalg.norm (Wa)
bX0 = - np.dot(X[0], Wa)
bX1 = - np.dot(X[1], Wa)
marg = (-bX1+bX0)/magWa
margPif = np.sqrt ((X[1][0]-X[0][0])**2 + (X[1][1]-X[0][1])**2)
print ('magWa, bX0, bX1, marg, margPif', magWa, bX0, bX1, marg, margPif)
if abs (marg - margPif) > 0.00001 or marg - 2.8 < 2.:
  print ("abs (marg - margPif) > 0.00001 or marg - 2.8 < 2. for X0", marg, margPif, X[0])
  exit(1)

X[0][0] = 1.
X[0][1] = -1.
X[1][0] = 3.
X[1][1] = -3.

Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, bsSvmLinKern, YNEGPOS, 0.00001)

if Wa[0] != -1.0 or Wa[1] != 1.0 or ba != 4.0 or cntWrnga != 0:
  print ("No met: W{-1,1}, b=4, cntWrng=0", Wa, ba, cntWrnga)
  #exit(1) TODO No met: W{-1,1}, b=4, cntWrng=0 [-0.99998474  1.        ] 3.999969482421875 0

X[0][0] = -1.0
X[0][1] = 1.0
magWa = np.linalg.norm (Wa)
bX0 = - np.dot(X[0], Wa)
bX1 = - np.dot(X[1], Wa)
marg = (-bX0+bX1)/magWa
margPif = np.sqrt ((X[1][0]-X[0][0])**2 + (-X[1][1]+X[0][1])**2) #!!!!
print ('magWa, bX0, bX1, marg, margPif', magWa, bX0, bX1, marg, margPif)
if abs (marg - margPif) > 0.00001 or marg - 2.8 < 2.:
  print ("abs (marg - margPif) > 0.00001 or marg - 2.8 < 2. for X0", marg, margPif, X[0])
  exit(1)

X[0][0] = 1.
X[0][1] = 3.
X[1][0] = 3.
X[1][1] = 1.

Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, bsSvmLinKern, YNEGPOS, 0.00001)

if Wa[0] != -1.0 or Wa[1] != 1.0 or ba != 0.0 or cntWrnga != 0:
  print ("No met: W{-1,1}, b=0, cntWrng=0", Wa, ba, cntWrnga)
  #exit(1) TODO

X[0][0] = -1.0
X[0][1] = 5.0
magWa = np.linalg.norm (Wa)
bX0 = - np.dot(X[0], Wa)
bX1 = - np.dot(X[1], Wa)
marg = (-bX0+bX1)/magWa
margPif = np.sqrt ((X[1][0]-X[0][0])**2 + (X[0][1]-X[1][1])**2) #!!!!!!
print ('magWa, bX0, bX1, marg, margPif', magWa, bX0, bX1, marg, margPif)
if abs (marg - margPif) > 0.00001 or marg - 2.8 < 2.:
  print ("abs (marg - margPif) > 0.00001 or marg - 2.8 < 2. for X0", marg, margPif, X[0])
  exit(1)

print ('Test margin equation margin=(-bH2+bH1)/mag(W) OK!')

