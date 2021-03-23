#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
from BsLibMisc import *
from dig012l import *
import numpy as np

#finding alphas and the b by solving linear system equation according to MIT MIT6_034F10_tutor05.pdf page 6 (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)
#error in K(C, A) = 2*0+0*0 = 2
LP = np.array ([[-1.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, -2.0, 2.0, 1.0], [0.0, -2.0, 4.0, 1.0]])
RP = np.array ([0.0, -1.0, -1.0, 1.0])
ALB = np.linalg.solve (LP, RP)
print ('MIT a0, a1, a3, b: ', ALB) #MIT a0, a1, a3, b:  [-0.  1.  1. -1.]

NUMS = mkDig123 ()

#64 points in 64D
NUMS.shape = (digCnt, smpCnt)
X = np.zeros ((2, smpCnt))
Y = np.zeros ((2), dtype=np.int16)
kern = BsSvmLinKern ()
#0-1
X[0] = NUMS[0]
X[1] = NUMS[1]
Y[0] = -1
Y[1] = 1
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP = np.array ([[-1.0, 1.0, 0.0], [k00, k10, 1.0], [k01, k11, 1.0]])
RP = np.array ([0.0, -1.0, 1.0])
print ('LP:\n', LP)
print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

#bsPrnImgTxt (X[0], 0, digSz)
#bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('0-1 margin=2/|W|: ', margStd)

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
print ('b0: ',  b0)
print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
print ('margin=(-b1-b0)/|W|: ', margB12)

#0-2
X[1] = NUMS[2]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

#bsPrnImgTxt (X[0], 0, digSz)
#bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('0-2 margin=2/|W|: ', margStd)

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1) #margB12 != margStd:  1.1547005383792528 1.1547005383792512

#1-2
X[0] = NUMS[1]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

#bsPrnImgTxt (X[0], 0, digSz)
#bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('1-2 margin=2/|W|: ', margStd)

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

#Classification by minimal margin:

#test data:
NUMS.shape = (digCnt, digSz, digSz)
NUMSt = mkDig123t (NUMS)
NUMS.shape = (digCnt, smpCnt)
NUMSt.shape = (digCnt, smpCnt)

#0t-0
X[0] = NUMSt[0]
X[1] = NUMS[0]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag

margmin = margStd
marg0t0 = margStd

print ('0t-0 margin=2/|W|: ', margStd)

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

#0t-1
X[1] = NUMS[1]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('0t-1 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

#0t-2
X[1] = NUMS[2]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('0t-2 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

if margmin != marg0t0:
  print ('margmin != marg0t0', margmin, marg0t0)
  exit (1)


#2t-0
X[0] = NUMSt[2]
X[1] = NUMS[0]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag

print ('2t-0 margin=2/|W|: ', margStd)
margmin = margStd

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

#2t-1
X[1] = NUMS[1]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
print ('2t-1 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

#2t-2
X[1] = NUMS[2]
k00 = kern.dot (X[0], X[0])
k01 = kern.dot (X[0], X[1])
k10 = kern.dot (X[1], X[0])
k11 = kern.dot (X[1], X[1])

LP[1][0] = k00
LP[1][1] = k10
LP[2][0] = k01
LP[2][1] = k11

#print ('LP:\n', LP)
#print ('RP:\n', RP)
ALB = np.linalg.solve (LP, RP)

bsPrnImgTxt (X[0], 0, digSz)
bsPrnImgTxt (X[1], 0, digSz)
print ('ALB:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[0]*ALB[0]*X[0] + Y[1]*ALB[1]*X[1]
print ('W:\n', W)
wMag = np.linalg.norm (W) #np.sqrt(x.dot(x))
margStd = 2.0/wMag
marg2t2 = margStd
margStd
print ('2t-2 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

b0 = - kern.dot (X[0], W)
b1 = - kern.dot (X[1], W)
#print ('b0: ',  b0)
#print ('b1: ',  b1)
margB12 = (-b1-b0)/wMag
if abs (margB12 - margStd) > 0.00001:
  print ('abs (margB12 - margStd) > 0.00001: ', margB12, margStd)
  exit (1)

if margmin != marg2t2:
  print ('margmin != marg2t2', margmin, marg2t2)
  exit (1) #margmin != marg2t2 0.5345224838248488 1.4142135623730954 - 2t still close to 0!

print ('Test OK!')

