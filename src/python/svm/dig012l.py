#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import numpy as np

#reusable lib

digCnt = 3
digSz = 8
smpCnt = digSz * digSz

def mkDig123 ():
  #array for three digits 0, 1, 2:
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
  return NUMS

def mkDig123t (pNUMS):
  NUMS = pNUMS.copy()
  NUMS[0][3][3] = 1
  NUMS[0][3][4] = 1
  NUMS[0][4][3] = 1
  NUMS[0][4][4] = 1
  for i in range (digSz - 2):
    NUMS[1][i+1][2] = 1
    NUMS[1][i+1][4] = 0
  NUMS[1][2][1] = 1
  NUMS[1][3][1] = 1
  NUMS[2][3][3] = 1
  NUMS[2][3][4] = 1
  NUMS[2][4][3] = 1
  NUMS[2][4][4] = 1
  return NUMS
