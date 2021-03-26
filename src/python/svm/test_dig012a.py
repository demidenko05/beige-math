#!/usr/bin/env python
# coding=UTF-8

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/..']
from BsLibSvm import *
from BsLibMisc import *
from dig012l import *
import numpy as np

#finding alphas and the b by solving linear system equation according to MIT MIT6_034F10_tutor05.pdf page 6 (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)
#author Yury Demidenko

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
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

LP = np.array ([[-1.0, 1.0, 0.0], [Y[0]*k00, Y[1]*k10, 1.0], [Y[0]*k01, Y[1]*k11, 1.0]])
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

#normalize W to [-1,1]:
mx = sys.float_info.min
for i in range (W.shape[0]):
  wi = abs (W[i])
  if wi > mx:
    mx = wi
nc = 1.0 / mx
Wn = W.copy ()
for i in range (Wn.shape[0]):
  Wn[i] *= nc
print ('Wn[-1:1]:', Wn)
#wMagn = np.linalg.norm (W)
wMagn = np.sqrt (np.dot (Wn, Wn))
bX0 = - np.dot(X[0], Wn)
bX1 = - np.dot(X[1], Wn)
margn = (-bX0+bX1)/wMagn
print ('wMagn, bX0, bX1, margn', wMagn, bX0, bX1, margn)
#wMagn, bX0, bX1, margn 4.898979485566356 -16.0 8.0 4.898979485566357 vs 1.6329931618554523
#but margn = abs(bX0+bX1)/wMagn = 8/4.898979485566356 = 1.632993162
#i.e. wrong margin equation for normalized W?! TODO TODO TODO TODO TODO

#0-2
X[1] = NUMS[2]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#1-2
X[0] = NUMS[1]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#Classification by minimal margin:

#test data:
NUMS.shape = (digCnt, digSz, digSz)
NUMSt = mkDig123t (NUMS)
NUMS.shape = (digCnt, smpCnt)
NUMSt.shape = (digCnt, smpCnt)

#0t-0
X[0] = NUMSt[0]
X[1] = NUMS[0]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#0t-1
X[1] = NUMS[1]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#0t-2
X[1] = NUMS[2]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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


if margmin != marg0t0:
  print ('margmin != marg0t0', margmin, marg0t0)
  exit (1)


#2t-0
X[0] = NUMSt[2]
X[1] = NUMS[0]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#2t-1
X[1] = NUMS[1]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

#2t-2
X[1] = NUMS[2]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

print ('2t-2 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

if margmin != marg2t2:
  print ('margmin != marg2t2', margmin, marg2t2)
  exit (1) #OK

#1t-0
X[0] = NUMSt[1]
X[1] = NUMS[0]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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

print ('1t-0 margin=2/|W|: ', margStd)
margmin = margStd

#1t-1
X[1] = NUMS[1]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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
marg1t1 = margStd
print ('1t-1 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

#2t-2
X[1] = NUMS[2]
k00 = Y[0] * kern.dot (X[0], X[0])
k01 = Y[0] * kern.dot (X[0], X[1])
k10 = Y[1] * kern.dot (X[1], X[0])
k11 = Y[1] * kern.dot (X[1], X[1])

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
print ('1t-2 margin=2/|W|: ', margStd)
if margmin > margStd:
  margmin = margStd

if margmin != marg1t1:
  print ('margmin != marg1t1', margmin, marg1t1)
  exit (1)

print ('Test OK!')

