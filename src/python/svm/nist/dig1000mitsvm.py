#!/usr/bin/env python
# coding=UTF-8
# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder
# Finding alphas and the b by solving linear system equation according to MIT MIT6_034F10_tutor05.pdf page 6 (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)

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
  for iPix in range (pixCntAll):
    if NUMS[d][iPix] > 0:
      NUMS[d][iPix] = 1

#kern = BsSvmPolyKern (1.0, 9.0)
#kern = BsSvmRbfKern (1.0/pixCnt)
kern = BsSvmLinKern ()
X = np.zeros ((2, pixCnt))
Y = np.array ([-1, 1])
LP = np.array ([[-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
RP = np.array ([0.0, Y[0], Y[1]])


for dTst in range (10):
#  dTst = 4 #debug
  for iTst in range (testCnt):
    margminDigIdx = [sys.float_info.max, -1, 0] #minmarg, train dig, train dig idx
    for iPix in range (pixCnt):
      X[0][iPix] = NUMS[dTst][testOfst + iTst*pixCnt + iPix]
    for dTrn in range (10):
      for iTrn in range (trainCnt):
        for iPix in range (pixCnt):
          X[1][iPix] = NUMS[dTrn][iTrn*pixCnt + iPix]
        LP[1][0] = Y[0] * kern.dot (X[0], X[0])
        LP[1][1] = Y[1] * kern.dot (X[1], X[0])
        LP[2][0] = Y[0] * kern.dot (X[0], X[1])
        LP[2][1] = Y[1] * kern.dot (X[1], X[1])
        try:
          ALB = np.linalg.solve (LP, RP)
          if not ( np.allclose (np.dot (LP, ALB), RP) ): #useless 2dots!
            print ('No met: np.allclose (np.dot (LP, ALB), RP)')
          W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
          wMag = np.linalg.norm (W)
          marg = 2.0/wMag
          if marg < margminDigIdx[0]:
            margminDigIdx[0] = marg
            margminDigIdx[1] = dTrn
            margminDigIdx[2] = iTrn
        except:
          #numpy.linalg.LinAlgError: Singular matrix - identical scaled digits - e.g. Error on dTst, iTst, dTrn, iTrn:  1 72 1 205
          #TODO if singular, then absolutely matched i.e. stop iteration
          print ('Error on dTst, iTst, dTrn, iTrn: ', dTst, iTst, dTrn, iTrn)
          print ('Digit test:')
          bsPrnImgTxt (X[0], 0, digSz)
          print ('Digit train:')
          bsPrnImgTxt (X[1], 0, digSz)
          print (sys.exc_info ()[0])
    if margminDigIdx[1] != dTst:
      wrongCnt += 1
    if margminDigIdx[1] == -1:
      print ('Can not classify dTst, iTst: ', dTst, iTst)
    else:
      PRED[dTst * testCnt + iTst] = margminDigIdx[1]
      print ('test digit, test index, minimum: train digit, train index, margin: ', dTst, iTst, margminDigIdx[1], margminDigIdx[2], margminDigIdx[0])
      print ('Digit test:')
      bsPrnImgTxt (NUMS[dTst], testOfst + iTst*pixCnt, digSz)
      print ('Digit train:')
      bsPrnImgTxt (NUMS[margminDigIdx[1]], margminDigIdx[2]*pixCnt, digSz)


ftest = open (pth+'/nist'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'t.mpred', 'w')

accur =  (testCntTot - wrongCnt) / testCntTot * 100.0
ftest.write ('Accuracy = ' + str (accur) + '%\n')

for d in range (10):
  for j in range (testCnt):
    ftest.write (' ' + str (PRED[d * testCnt + j]))
  ftest.write ('\n')

ftest.close ()
