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

#test data form the first article calc.py:
X = np.array ([[.5, 1.2], [1., 1.], [3., 3.], [4., 3.3]])
Y = np.array ([0, 0, 1, 1])
W = np.array ([1., 0.])
Wm = W.copy ()
YNEGPOS = bsSvmCheckData (X, Y, W, bsSvmLinKern)
MARGB12m = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
W[1] = 1.
MARGB12 = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
W[0] = 0.
MARGB12 = bsSvmFndMinMarg (X, Y, W, bsSvmLinKern, YNEGPOS)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy ()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = int(MARGB12m[3]) + 1
j = int(MARGB12m[4]) + 1
err = 0
if b != -4.:
  err = 1
  print ("b != -4! b = %s " % b)

if not( i == 2 and j == 3) or not( j == 3 and i == 2):
  err = 1
  print ("i, j != 2, 3! - ", i , j)

if err == 0:
  print ('Test OK!')
