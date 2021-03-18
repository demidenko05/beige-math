#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import numpy as np
import matplotlib.pyplot as plt
from dig012l import *

NUMS = mkDig123 ()

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

NUMS.shape = (digCnt, digSz, digSz)
NUMSt = mkDig123t (NUMS)
#test 0 modified:
plt.subplot (2, digCnt, 4)
plt.axis ('off')
plt.imshow (NUMSt[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 0')
#test 1 modified:
plt.subplot (2, digCnt, 5)
plt.axis ('off')
plt.imshow (NUMSt[1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 1')
#test 2 modified:
plt.subplot (2, digCnt, 6)
plt.axis ('off')
plt.imshow (NUMSt[2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title ('Test 2')

NUMSt.shape = (digCnt, smpCnt)
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-0.t', 'w')
f.write(str(0))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMSt[0][j]))
f.write('\n')
f.close()
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-1.t', 'w')
f.write(str(1))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMSt[1][j]))
f.write('\n')
f.close()
f = open('dig'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz)+'-2.t', 'w')
f.write(str(2))
for j in range (smpCnt):
  f.write(' ' + str(j+1) + ':' + str(NUMSt[2][j]))
f.write('\n')
f.close()
plt.show ()
