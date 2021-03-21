#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#Classify NIST data - train (900 samples) and test (the rest 100) files
#NIST data from http://www.cis.jhu.edu/~sachin/digit/digit.html 1000 28x28 digits (unsigned char 8bit): data0..data9
#required path to the data in command line sys.argv[1]

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/../..']
from BsLibSvm import *
from BsLibMisc import *
import numpy as np

def prnUsage ():
  print ('You must pass path to the NIST files 1000 28x28 digits (unsigned char 8bit): data0..data9!')
  print ('from http://www.cis.jhu.edu/~sachin/digit/digit.html')
  print ('Use: python dig1000bssvm.py [path_to_nist_files]')

ip = 0
pth = ''
for arg in sys.argv:
  if ip == 1:
    pth = arg
  ip += 1

if pth == '' or pth == '-h':
  prnUsage ()
  exit (1)

digCnt = 1000
digSz = 28
pixCnt = digSz * digSz
pixCntAll = digCnt * pixCnt

trainCnt = 900
testCnt = digCnt - trainCnt
testOfst = trainCnt * pixCnt

testCntTot = testCnt * 10

wrongCnt = 0

PRED = np.zeros ((testCntTot), dtype=np.uint8)

NUMS = [0,1,2,3,4,5,6,7,8,9]

try:

  for d in range (10):

    fnme = pth + "/data" + str (d)
    NUMS[d] = np.fromfile (fnme, dtype=np.uint8)

    if NUMS[d].shape[0] != pixCntAll:
      print ('It must be 1000 uint8 28x28 samples in ', fnme)
      raise
      
    #just for visual control, print several samples:
    print ("Samples #0,1,2,900,901,902 From file: ", fnme)
    bsPrnImgTxt (NUMS[d], 0, digSz)
    bsPrnImgTxt (NUMS[d], 1*pixCnt, digSz)
    bsPrnImgTxt (NUMS[d], 2*pixCnt, digSz)
    bsPrnImgTxt (NUMS[d], 900*pixCnt, digSz)
    bsPrnImgTxt (NUMS[d], 901*pixCnt, digSz)
    bsPrnImgTxt (NUMS[d], 902*pixCnt, digSz)
    #there is 5 #900 that looks like 6


except:
  prnUsage ()
  print (sys.exc_info ()[0])
  raise

#scaling to 0-1
for d in range (10):
  for ipx in range (pixCntAll):
    if NUMS[d][ipx] > 0:
      NUMS[d][ipx] = 1

#kern = BsSvmPolyKern (1.0, 9.0)
kern = BsSvmRbfKern (1.0/pixCnt)
X = np.zeros ((2, pixCnt))
Y = np.zeros ((2), dtype=np.int16)
Y[1] = 1
minStp = 0.001
YNEGPOS = False
for d in range (10):
  it = 0
  for i in range (testCnt):
    for ipx in range (pixCnt):
      X[0][ipx] = NUMS[d][testOfst + i*pixCnt + ipx]
    for j in range (trainCnt):
      for ipx in range (pixCnt):
        X[1][ipx] = NUMS[d][j*pixCnt + ipx]
      if it == 0:
        YNEGPOS = bsSvmCheckData (X, Y, kern)
      Wa, ba, cntWrnga, marga = bsSvmTrain (X, Y, kern, YNEGPOS, minStp)
      if it < 2:
        print ('Digit test:')
        bsPrnImgTxt (X[0], 0, digSz)
        print ('Digit train:')
        bsPrnImgTxt (X[1], 0, digSz)
        it += 1
#non-scaling poly 9
#0-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [4.57884289e+39 -2.82485959e+38 -1.15210584e+41] 7840 0 0 0 1
#0-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.40568112e+37 4.13577278e+38 5.86868106e+37] 7840 0 0 0 1
#scaling 0-1 poly 9
#0-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.91764603e+17 1.46320679e+19 9.71335721e+18] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.08844573e+20 -1.04308405e+19 -2.74254261e+21] 7840 0 0 0 1
#0-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [6.32187058e+17 -7.60231059e+17 -1.70014164e+19] 7840 0 0 1 0
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [6.60399031e+17 -3.54520878e+16 -1.70014164e+19] 7840 0 0 1 0
#0-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.19722044e+19 -5.61752833e+18 -3.07720364e+20] 7840 0 0 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [6.16342828e+19 -1.70014164e+19 -1.60041537e+21] 7840 0 0 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.27228745e+17 -8.59475475e+18 -1.70014164e+19] 7840 0 0 1 0
#scaling 0-1 RBF:
#0-2 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [7.38265432e-110 -3.14727032e-144 -1.85156115e-108] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.78136532e-106 -7.92801133e-153 -4.57641200e-105] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.01857140e-101 -2.29788796e-145 -7.63048368e-100] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [2.73135996e-142 -7.92801133e-153 -7.01699327e-141] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [8.93280633e-130 -9.65098252e-171 -2.34645687e-128] 7840 0 1 0 1
#0-1 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.08597313e-154 -6.18540508e-177 -7.92801133e-153] 7840 0 1 1 0
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.08597313e-154 -7.26433219e-195 -7.92801133e-153] 7840 0 1 1 0
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.08597313e-154 -2.66176079e-190 -7.92801133e-153] 7840 0 1 1 0
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [3.08597313e-154 -1.46541764e-177 -7.92801133e-153] 7840 0 1 1 0
#0-0 MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [7.94961201e-118 -1.45103947e-154 -2.03454240e-116] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [8.24965375e-113 -1.14570372e-143 -2.07064762e-111] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [7.93757358e-118 -1.43589397e-154 -2.03456050e-116] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [1.96196182e-089 -4.16017247e-143 -4.92839412e-088] 7840 0 1 0 1
#    MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m:  [4.51573110e-097 -5.47112966e-142 -1.13613548e-095] 7840 0 1 0 1
# conclusion - doesn't work and slow - transforming into pixels count dimension plus transformation


# ftest = open (pth+'/nist'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'t.pred', 'w')

# accur =  (testCntTot - wrongCnt) / testCntTot * 100.0
# ftest.write ('Accuracy = ' + str (accur) + '%\n')

# for d in range (10):
  # for j in range (testCnt):
    # ftest.write (' ' + str (PRED[d * testCnt + j]))
  # ftest.write ('\n')

# ftest.close ()
