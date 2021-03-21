#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#Python/NumPy based SVM library
#see tests and usages in svm folder

import sys
import numpy as np
import math

#Validate data
#pXARR - array(number of samples, dimension), type should by float64
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pKernel - kernel with dot function for two vectors
#return exception or array(2)  0 - negative Y, 1 - positive Y
def bsSvmCheckData (pXARR, pYARR, pKernel):
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
#pKernel - kernel with dot function for two vectors
#pYNEGPOS - array(2) 0 - negative Y, 1 - positive Y
#return exception if error or [array(3) float64, int, int, int] - [minimum margin, bH1, bH2], i1, i2, countWrong
def bsSvmFndMinMarg (pXARR, pYARR, pWVEC, pKernel, pYNEGPOS):
  # find [minimal margin, bH1, bH2], i1, i2, countWrong:
  MARGB12 = np.zeros(3)
  i1 = 0
  i2 = 0
  cntWrng = 0
  #1. compute all b[i] and data for wrong data decision:
  bMinMax = np.zeros (8)
  bMin = -sys.float_info.max #-1.7976931348623157e+308
  bMax = sys.float_info.max #1.7976931348623157e+308
  bInfp = float ('Infinity')
  bInfn = float ('-Infinity')
  bNan = float ('nan')
  bMinMax[0] = bMax
  bMinMax[1] = bMin
  bMinMax[2] = bMax
  bMinMax[3] = bMin
  sz = pYARR.shape[0]
  B = np.zeros (sz)
  for i in range (sz):
    #for 2D: x1w1 + x2w2 + b = 0 -> b = -x1w1 -x2w2
    #here b = -K(X,W)
    B[i] = - pKernel.dot (pXARR[i], pWVEC)
    if B[i] == bInfp or B[i] == bInfn or B[i] == bNan or B[i] == bMin or B[i] == bMax:
      print ('Not yet implemented: Bi: ',  B[i])
      raise
    if pYARR[i] == pYNEGPOS[0]:
      if B[i] < bMinMax[0]:
        bMinMax[0] = B[i]
        MARGB12[1] = B[i]
        i1 = i
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
        i2 = i

  #if there is a wrong sample/s then fix mini-maxes:
  if ( bMinMax[0] >= bMinMax[2] and bMinMax[0] <= bMinMax[3] ) or ( bMinMax[1] >= bMinMax[2] and bMinMax[1] <= bMinMax[3] ) or ( bMinMax[2] >= bMinMax[0] and bMinMax[2] <= bMinMax[1] ) or ( bMinMax[3] >= bMinMax[0] and bMinMax[3] <= bMinMax[1] ):
    #print('MINMAX: ', bMinMax)
    idxNeg = 0
    idxPos = 1
    midNeg = bMinMax[0] + (bMinMax[1] - bMinMax[0]) / 2.
    midPos = bMinMax[2] + (bMinMax[3] - bMinMax[2]) / 2.
    isInverse = 0
    if midNeg < midPos:
      isInverse = 1
    if isInverse == 1:
      MARGB12[1] = bMinMax[4]
      MARGB12[2] = bMinMax[6]
      i1 = int (bMinMax[5])
      i2 = int (bMinMax[7])
      idxNeg = 1
      idxPos = 0
      bmm = bMinMax[0]
      bMinMax[0] = bMinMax[2]
      bMinMax[2] = bmm
      bmm = bMinMax[1]
      bMinMax[1] = bMinMax[3]
      bMinMax[3] = bmm
      #print('MINMAXc: ', bMinMax)
    brdr = bMin
    if bMinMax[2] == bMinMax[3]:
      #dot 1 inside region 0
      bMinMax[0] = bMax
      brdr = bMinMax[3]
    elif bMinMax[0] == bMinMax[1]:
      #dot 0 inside region 1
      bMinMax[2] = bMin
      brdr = bMinMax[0]
    elif bMinMax[3] - bMinMax[2] > bMinMax[1] - bMinMax[0]:
      #region 1 inside region 0
      bMinMax[2] = bMin
      brdr = bMinMax[0]
    else:
      #region 0 inside region 1
      bMinMax[0] = bMax
      brdr = bMinMax[3]
    if bMinMax[0] == bMax:
      for i in range (sz):
        if pYARR[i] == pYNEGPOS[idxNeg]:
          if B[i] <= brdr:
            cntWrng += 1
          elif B[i] < bMinMax[0]:
            bMinMax[0] = B[i]
            MARGB12[1] = B[i]
            i1 = i
      if bMinMax[0] == bMax:
        #tst3d1.py
        #MINMAX[-2. -2. -2. -2. -2.  3. -2.  0.]
        #W:  [0. 1. 1.]
        #B:  [-2. -2. -2. -2. -2. -2.]
        #brdr:  -2.0
        #cntWrng:  3 - all
        bMinMax[0] = bMinMax[1]
        MARGB12[1] = bMinMax[1]
        # i don't care it's wrong hyperplane
    elif bMinMax[2] == bMin:
      for i in range (sz):
        if pYARR[i] == pYNEGPOS[idxPos]:
          if B[i] >= brdr:
            cntWrng += 1
          elif B[i] > bMinMax[2]:
            bMinMax[2] = B[i]
            MARGB12[2] = B[i]
            i2 = i
      if bMinMax[2] == bMin:
        bMinMax[2] = bMinMax[3]
        MARGB12[2] = bMinMax[3]

  elif bMinMax[0] < bMinMax[2]:
    MARGB12[1] = bMinMax[4]
    MARGB12[2] = bMinMax[6]
    i1 = int (bMinMax[5])
    i2 = int (bMinMax[7])

  magW = 0.0
  for i in range (pWVEC.shape[0]):
    magW += pWVEC[i]**2
  magW = np.sqrt (magW)

  MARGB12[0] = (- MARGB12[2] + MARGB12[1]) / magW
  if MARGB12[0] < 0.0:
    print ('Wrong algorithm:\n  MINMAX: ', bMinMax)
    print ('  cntWrng: ', cntWrng)
    print ('  B: ', B)
    print ('  W: ', pWVEC)
    print ('  MARGB12: ', MARGB12)
    raise
  return [MARGB12, i1, i2, cntWrng]

#Linear kernel
#passed data must be preliminary validated! 
#return dot product of two vectors
class BsSvmLinKern:
  def dot (self, pVEC1, pVEC2):
    rz = 0. #TODO float64
    for i in range (pVEC1.shape[0]):
      rz += pVEC1[i] * pVEC2[i]
    return rz

#Find separating hyperplane - coefficient vector W and shifting b, also returns count of non-separated samples
#pXARR - array(number of samples, dimension), type should by float64
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pKernel - kernel with dot function for two vectors
#pYNEGPOS - array(2) 0 - negative Y, 1 - positive Y
#pMinStep - minimum step to rotate W
#return [array(dimension) float64, float64, int32, float64] - coefficient vector W and shifting b, also returns count of non-separated samples and the margin
def bsSvmTrain (pXARR, pYARR, pKernel, pYNEGPOS,  pMinStep):
  dimns = pXARR.shape[1]
  W = np.ones (dimns)
  Wm = W.copy ()
  MARGB12m, i1m, i2m, cntWrngm = bsSvmFndMinMarg (pXARR, pYARR, W, pKernel, pYNEGPOS)
  cntWrngMax = cntWrngm
  cntWrngMin = cntWrngm
  cntItr = 0
  for i in range (dimns):
    #swing left - right, next step = 1/2 previous step, i.e. W[i]= -1 -1; -0.5 +0.5, ...
    stp = 1.0
    while True:
      cntItr += 1

      if Wm[i] != -1.0:
        W[i] -= stp
      else:
        W[i] += stp

      MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (pXARR, pYARR, W, pKernel, pYNEGPOS)
      if cntWrng > cntWrngMax:
        cntWrngMax = cntWrng
      if cntWrng < cntWrngMax:
        cntWrngMin = cntWrng

      if MARGB12[0] > MARGB12m[0]:
        MARGB12m = MARGB12
        Wm[i] = W[i]
        i1m = i1
        i2m = i2
        cntWrngm = cntWrng
      elif not ( ( Wm[i] == 1.0 and stp < 1.0 ) or Wm[i] == -1.0 ):
        if Wm[i] == 1.0:
          W[i] = Wm[i] - stp
        else:
          W[i] = Wm[i] + stp
        MARGB12, i1, i2, cntWrng = bsSvmFndMinMarg (pXARR, pYARR, W, pKernel, pYNEGPOS)
        if cntWrng > cntWrngMax:
          cntWrngMax = cntWrng
        if cntWrng < cntWrngMax:
          cntWrngMin = cntWrng
        if MARGB12[0] > MARGB12m[0]:
          Wm[i] = W[i]
          MARGB12m = MARGB12
          i1m = i1
          i2m = i2
          cntWrngm = cntWrng

      stp = stp / 2.0
      if stp < pMinStep:
        break

  print ('MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m: ', MARGB12m, cntItr, cntWrngMin, cntWrngMax, i1m, i2m)

  b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)

  return [Wm, b, cntWrngm, MARGB12m[0]]

#RBF kernel
class BsSvmRbfKern:

  def __init__(self, pGamma):
    self.gamma = pGamma

  #passed data must be preliminary validated! 
  #return dot product of two vectors
  def dot (self, pVEC1, pVEC2):
    SP = pVEC1 - pVEC2
    mag = 0.0 #TODO float64
    for i in range (SP.shape[0]):
      mag += SP[i] * SP[i]
    mag2 = mag * mag
    return math.exp(-self.gamma*mag2)

#Polynomial kernel
class BsSvmPolyKern:

  def __init__(self, pB, pN):
    self.b = pB
    self.n = pN

  #passed data must be preliminary validated! 
  #return dot product of two vectors
  def dot (self, pVEC1, pVEC2):
    sm = 0.
    for i in range (pVEC1.shape[0]):
      sm += pVEC1[i] * pVEC2[i]
    return (sm + self.b) ** self.n
