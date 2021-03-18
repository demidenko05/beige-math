#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
from LIBSVMORIG import *
from dig012l import *
import numpy as np

NUMS = mkDig123 ()

#64 points in 64D
NUMS.shape = (digCnt, smpCnt)
X = np.zeros ((2, smpCnt))
Y = np.zeros ((2), dtype=np.int16)
kern = BsSvmLinKern ()
minStp = 0.0001
#0-1
X[0] = NUMS[0].copy()
X[1] = NUMS[1].copy()
Y[0] = 0
Y[1] = 1
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
#0-2
X[0] = NUMS[0].copy()
X[1] = NUMS[2].copy()
Y[0] = 0
Y[1] = 1
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
#1-2
X[0] = NUMS[1].copy()
X[1] = NUMS[2].copy()
Y[0] = 0
Y[1] = 1
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
#results:
#0-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.17953565 19.99755859 11.99853516] 896 0 0 0 1
#0-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.73029674 7.99755859 3.99804688] 896 0 0 0 1
#1-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.64888568 15.99804688 11.99853516] 896 0 0 1 0

#test data:
NUMS.shape = (digCnt, digSz, digSz)
NUMSt = mkDig123t (NUMS)
NUMS.shape = (digCnt, smpCnt)
NUMSt.shape = (digCnt, smpCnt)

#test 0t:
#0t-0
X[0] = NUMSt[0].copy()
X[1] = NUMS[0].copy()
Y[0] = 0
Y[1] = 1
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg0t0 = marga
margmin = marga
#0t-1
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#0t-2
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#results
#0t-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.26491101 5.99707031 1.99755859] 896 0 0 0 1
#0t-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.76930347 23.99707031 11.99853516] 896 0 0 0 1
#0t-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.46059347 11.99707031  3.99804688] 896 0 0 0 1
if margmin != marg0t0:
  print ('ERR: marg0t0 != margmin', marg0t0, margmin)
  exit (1)
print ('Test linear 0t OK!')

#test 1t:
#1t-1
X[0] = NUMSt[1].copy()
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg1t1 = marga
margmin = marga
#1t-0
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#1t-2
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
if margmin != marg1t1:
  print ('ERR: marg1t1 != margmin', marg1t1, margmin)
  exit (1)
print ('Test linear 1t OK!')

#test 2t linear:
#2t-2
X[0] = NUMSt[2].copy()
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
print ('2t2 X:\n', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-2:\n  marga, ba: ', marga, ba)
#print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg2t2 = marga
margmin = marga
#2t-0
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
print ('2t0 X:\n', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#2t-1
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#results:
#2t-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.63245551 3.99780273 1.99804688] 896 0 0 0 1
#2t-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.39223227 7.99755859 5.99780273] 896 0 1 1 0
#2t-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.88465173 17.99780273 11.99853516] 896 0 1 0 1
#so, 2t is close to 0 for linear kernel and any minStp!!!
if margmin != marg2t2:
  print ('ERR: linear kernel marg2t2 != margmin', marg2t2, margmin)
  #exit (1)
else:
  print ('Test linear 2t OK!')

#test 2t RBF LIBSVM:
kern = LibSvmOrigKernRbf (1.0/smpCnt)
#2t-2
X[0] = NUMSt[2].copy()
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg2t2 = marga
margmin = marga
#2t-0
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#2t-1
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#results:
#2t-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.00447608 -0.18593541 -0.21069082] 896 0 1 0 1
#2t-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.00150228 -0.08739736 -0.09903343] 896 0 1 1 0
#2t-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [0.00450526 -0.12132765 -0.15578541] 896 0 1 0 1
#so, 2t is close to 0 for RBF-LIBSVM kernel for any minStp!!!

if margmin != marg2t2:
  print ('ERR: RBF-LIBSVM kernel marg2t2 != margmin', marg2t2, margmin)
  #exit (1)
else:
  print ('Test RBF SVMLIB 2t OK!')

#test 2t RBF:
kern = BsSvmRbfKern (1.0/smpCnt)
#2t-2
X[0] = NUMSt[2].copy()
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg2t2 = marga
margmin = marga
#2t-0
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#2t-1
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 2t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga

if margmin != marg2t2:
  print ('ERR RBF: RBF-LIBSVM kernel marg2t2 != margmin', marg2t2, margmin)
  exit (1)

print ('Test RBF 2t OK!')

#test 0t RBF:
#0t-0
X[0] = NUMSt[0].copy()
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg0t0 = marga
margmin = marga
#0t-1
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#0t-2
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 0t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
if margmin != marg0t0:
  print ('ERR RBF: marg0t0 != margmin', marg0t0, margmin)
  exit (1)
print ('Test RBF 0t OK!')

#test 1t:
#1t-1
X[0] = NUMSt[1].copy()
X[1] = NUMS[1].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-1:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
marg1t1 = marga
margmin = marga
#1t-0
X[1] = NUMS[0].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-0:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
#1t-2
X[1] = NUMS[2].copy()
YNEGPOS = bsSvmCheckData (X, Y, kern)
#print ('X: ', X)
#print ('Y: ', Y)
Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
print ('for 1t-2:\n  marga, ba: ', marga, ba)
print ('  Wa: ', Wa)
print ('  cntWrnga: ', cntWrnga)
if marga < margmin:
  margmin = marga
if margmin != marg1t1:
  print ('ERR RBF: marg1t1 != margmin', marg1t1, margmin)
  exit (1)
print ('Test RBF 1t OK!')
