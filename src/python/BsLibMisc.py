#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#MISC lib usually for tests purposes

import sys
import numpy as np
import cmath

#for tests purposes
#Inverse Y
#pYARR - array(number of samples) - correspondent separating function Y to samples [0,1] or [-1,1]
#pYNEGPOS - array(2) 0 - negative Y, 1 - positive Y
def bsSvmInverseY (pYARR, pYNEGPOS):
  yn = pYNEGPOS[0]
  yp = 1
  for i in range (pYARR.shape[0]):
    if pYARR[i] != yn:
      yp = pYARR[i]
      break
  for i in range (pYARR.shape[0]):
    if pYARR[i] == yn:
      pYARR[i] = yp
    else:
      pYARR[i] = yn

#Print image in text mode (in terminal)
#pROW - platen image row
#pStart - index start
#pSize - image size, e.g. 8, then it will print 64 values as 8x8 matrix
def bsPrnImgTxt (pROW, pStart, pSize):
  for i in range (pSize):
    for j in range (pSize):
      idx = pStart + (i * pSize) + j
      if pROW[idx] == 0.0:
        print (' ', end='')
      else:
        print ('*', end='')
    print ('')
