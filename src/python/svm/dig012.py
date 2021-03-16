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
import matplotlib.pyplot as plt

digCnt = 3
digSz = 8
smpCnt = digSz * digSz
#array for three digits 0, 1 , 2:
NUMS = np.zeros ((digCnt, digSz, digSz), dtype=np.int16)
for i in range (digSz - 2):
  NUMS[0][i+1][1] = 1
  NUMS[0][i+1][digSz-2] = 1
  NUMS[0][1][i+1] = 1
  NUMS[0][digSz-2][i+1] = 1
  NUMS[1][i+1][3] = 1
  NUMS[1][i+1][4] = 1
  NUMS[2][1][i+1] = 1
  NUMS[2][digSz-2][i+1] = 1
  NUMS[2][i+1][digSz-i-2] = 1
  
for i in range (digCnt):
  plt.subplot (2, digCnt, i+1)
  plt.axis ('off')
  plt.imshow (NUMS[i], cmap=plt.cm.gray_r, interpolation='nearest')
  plt.title ('%i' % i)
#save to LIBSVM formatted file:
NUMS.shape = (digCnt, smpCnt)
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz), 'w')
for i in range (digCnt):
  f.write(str(i))
  for j in range (smpCnt):
    f.write(' ' + str(j+1) + ':' + str(NUMS[i][j]))
  f.write('\n')
f.close()

#train by BsLibSvm:
#64 points in 64D
X = np.zeros ((smpCnt, smpCnt))
Y = np.zeros ((smpCnt), dtype=np.int16)
for i in range (smpCnt):
  X[i][i] = 1
  Y[i] = NUMS[1][i]

bsSvmRbfKern = BsSvmRbfKern (1.0/64.0)
YNEGPOS = bsSvmCheckData (X, Y, bsSvmRbfKern)
print ('X: ', X)
print ('Y: ', Y)
Wa, ba, cntWrnga = bsSvmTrain (X, Y, bsSvmRbfKern, YNEGPOS, 0.0001)
print ('for 1:\n  ba', ba)
print ('Wa: ', Wa)
print ('cntWrnga: ', cntWrnga)

#test 0 modified:
NUMS.shape = (digCnt, digSz, digSz)
NUMS[0][3][3] = 1
NUMS[0][3][4] = 1
NUMS[0][4][3] = 1
NUMS[0][4][4] = 1
plt.subplot (2, digCnt, 4)
plt.axis ('off')
plt.imshow (NUMS[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 0')
#test 1 modified:
for i in range (digSz - 2):
  NUMS[1][i+1][2] = 1
  NUMS[1][i+1][4] = 0
NUMS[1][2][1] = 1
NUMS[1][3][1] = 1
plt.subplot (2, digCnt, 5)
plt.axis ('off')
plt.imshow (NUMS[1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 1')
#test 2 modified:
NUMS[2][3][3] = 1
NUMS[2][3][4] = 1
NUMS[2][4][3] = 1
NUMS[2][4][4] = 1
plt.subplot (2, digCnt, 6)
plt.axis ('off')
plt.imshow (NUMS[2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 2')

NUMS.shape = (digCnt, smpCnt)
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-0.t', 'w')
f.write(str(0))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMS[0][j]))
f.write('\n')
f.close()
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-1.t', 'w')
f.write(str(1))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMS[1][j]))
f.write('\n')
f.close()
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-2.t', 'w')
f.write(str(2))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMS[2][j]))
f.write('\n')
f.close()
plt.show ()
