#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#Python/NumPy based SVM library
#see tests and usages in svm folder

import numpy as np

#Validate data
#pXARR - array(number of samples, dimension), type should by float64
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pWVEC - array(dimension) - coefficient vector
#pKernel - dot function for two vectors
#return exception or array(2)  0 - negative Y, 1 - positive Y
def bsSvmCheckData (pXARR, pYARR, pWVEC, pKernel):
  sz = pYARR.shape[0]
  if sz < 2:
    print ('Size of data less then 2!')
    raise
  #TODO comparing floats is usually problematic! check!
  YNEGPOS = np.zeros (2)
  YNEGPOS[0] = pYARR[0]
  YNEGPOS[1] = pYARR[0]
  for i in range (sz):
    if pYARR[i] != YNEGPOS[0]:
      YNEGPOS[1] = pYARR[i]
      break
  if YNEGPOS[0] == YNEGPOS[1]:
    print ('The only separating Y! Must be [0,1] or [-1,1]')
    raise 
  for i in range (sz):
    if pYARR[i] != YNEGPOS[0] and pYARR[i] != YNEGPOS[1]:
      print ('Only 2 separating Y supported! Must be [0,1] or [-1,1]!')
      raise
  #TODO farther data validation - arrays size, etc.
  return YNEGPOS

#Find minimum margin and correspondent samples indexes and [B]
#pXARR - array(number of samples, dimension), type should by float64
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pWVEC - array(dimension) - coefficient vector
#pKernel - dot function for two vectors
#pYNEGPOS - array(2) 0 - negative Y, 1 - positive Y
#return array(6) float64 - minimum margin, bH1, bH2, i1, i2, countWrong
def bsSvmFndMinMarg (pXARR, pYARR, pWVEC, pKernel, pYNEGPOS):
  # find minimal margin, bH1, bH2, i1, i2, countWrong:
  MARGB12 = np.zeros(6)
  #1. compute all b[i] and data for wrong data decision:
  idxNeg = 0
  idxPos = 1
  bMinMax = np.zeros (8)
  bMinMax[0] = 99999999999.99
  bMinMax[1] = -99999999999.99
  bMinMax[2] = 99999999999.99
  bMinMax[3] = -99999999999.99
  sz = pYARR.shape[0]
  B = np.zeros (sz)
  for i in range (sz):
    #for 2D: x1w1 + x2w2 + b = 0 -> b = -x1w1 -x2w2
    #here b = -K(X,W)
    B[i] = - pKernel (pXARR[i], pWVEC)
    if pYARR[i] == pYNEGPOS[idxNeg]:
      if B[i] < bMinMax[0]:
        bMinMax[0] = B[i]
        MARGB12[1] = B[i]
        MARGB12[3] = i
      if B[i] > bMinMax[1]:
        bMinMax[1] = B[i]
        bMinMax[6] = B[i]
        bMinMax[7] = i
    else:
      if B[i] < bMinMax[2]:
        bMinMax[2] = B[i]
        bMinMax[4] = B[i]
        bMinMax[5] = i
      if B[i] > bMinMax[3]:
        bMinMax[3] = B[i]
        MARGB12[2] = B[i]
        MARGB12[4] = i

#  print('MINMAX: ', bMinMax)

  #if there is a wrong sample/s then fix mini-maxes:
  if ( bMinMax[0] >= bMinMax[2] and bMinMax[0] <= bMinMax[3] ) or ( bMinMax[1] >= bMinMax[2] and bMinMax[1] <= bMinMax[3] ) or ( bMinMax[2] >= bMinMax[0] and bMinMax[2] <= bMinMax[1] ) or ( bMinMax[3] >= bMinMax[0] and bMinMax[3] <= bMinMax[1] ):
    print('MINMAX: ', bMinMax)
    if bMinMax[0] < bMinMax[2]:
      idxNeg = 1
      idxPos = 0
  elif bMinMax[0] < bMinMax[2]:
    MARGB12[1] = bMinMax[4]
    MARGB12[2] = bMinMax[6]
    MARGB12[3] = bMinMax[5]
    MARGB12[4] = bMinMax[7]

  magW = 0.
  for i in range (pWVEC.shape[0]):
    magW += pWVEC[i]**2
  magW = np.sqrt (magW)

  MARGB12[0]  = (- MARGB12[2] + MARGB12[1]) / magW
  return MARGB12

#Linear kernel
#passed data must be preliminary validated! 
#return dot product of two vectors
def bsSvmLinKern (pVEC1, pVEC2):
  rz = 0. #TODO float64
  for i in range (pVEC1.shape[0]):
    rz += pVEC1[i] * pVEC2[i]
  return rz


#for tests purposes
#Inverse Y
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pYNEGPOS - array(2) 0 - negative Y, 1 - positive Y
def bsSvmInverseY (pYARR, pYNEGPOS):
  for i in range (pYARR.shape[0]):
    if pYARR[i] == pYNEGPOS[0]:
      pYARR[i] = pYNEGPOS[1]
    else:
      pYARR[i] = pYNEGPOS[0]
