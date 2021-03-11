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
#return array(5) float64 - minimum margin, bH1, bH2, i1, i2
def bsSvmFndMinMarg (pXARR, pYARR, pWVEC, pKernel, pYNEGPOS):
  #1. compute all b[i]:
  sz = pYARR.shape[0]
  B = np.zeros (sz)
  for i in range (sz):
    #for 2D: x1w1 + x2w2 + b = 0 -> b = -x1w1 -x2w2
    #here b = -
    B[i] = - pKernel (pXARR[i], pWVEC)
  #2. find minimal margin:
  MARGB12 = np.zeros(5) #minimal  margin, bH1, bH2, i1, i2
  MARGB12[0] = 9999999999.99
  magW = 0.
  for i in range (pWVEC.shape[0]):
    magW += pWVEC[i]**2
  magW = np.sqrt(magW)
  for i in range (sz):
    if pYARR[i] == pYNEGPOS[0]:
      for j in range(sz):
        if i != j and pYARR[j] == pYNEGPOS[1]:
          marg = (- B[j] + B[i]) / magW
          if marg < MARGB12[0]:
            MARGB12[0] = marg
            MARGB12[1] = B[i]
            MARGB12[2] = B[j]
            MARGB12[3] = i
            MARGB12[4] = j
  return MARGB12

#Linear kernel
#passed data must be preliminary validated! 
#return dot product of two vectors
def bsSvmLinKern (pVEC1, pVEC2):
  rz = 0. #TODO float64
  for i in range (pVEC1.shape[0]):
    rz += pVEC1[i] * pVEC2[i]
  return rz
