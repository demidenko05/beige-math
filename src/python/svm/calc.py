#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import numpy as np
import matplotlib.pyplot as plt

def prnHp(pAx, pX, pY, pW, pSz, pLn):
  #1. compute all b[i] and draw support vectors:
  B = np.zeros(pSz)
  for i in range(pSz):
    #x1w1 + x2w2 + b = 0 -> b = -x1w1 -x2w2
    B[i] = - (pX[i][0]*pW[0]) - (pX[i][1]*pW[1])
    #pAx.annotate('b' + str(i) + '=' + str(B[i]), xy=(1.6, 1 + i/2.))
    #draw HP: x2=-x1w1/w2 - b then w1 and w2!=0, x2=-b then w1=0, x1=-b then w2=0
    x1a = 0.
    x1b = 0.
    x2a = 0.
    x2b = 0.
    if pW[0] == 0.:
      x1a = pX[i][0] - pLn
      x1b = pX[i][0] + pLn
      x2a = -B[i] 
      x2b = -B[i]
    elif pW[1] == 0.:
      x2a = pX[i][1] - pLn
      x2b = pX[i][1] + pLn
      x1a = -B[i] 
      x1b = -B[i]
    else:
      x1a = pX[i][0] - pLn
      x1b = pX[i][0] + pLn
      x2a = -(x1a * pW[0] / pW[1]) - B[i] 
      x2b = -(x1b * pW[0] / pW[1]) - B[i] 
    pAx.plot([x1a, x1b], [x2a, x2b], c="green")
  #2. find minimal margin:
  MARGB12 = np.zeros(5) #minimal  margin, bH1, bH2, i1, i2
  MARGB12[0] = 9999999999.99
  magW = np.sqrt(pW[0]**2. + pW[1]**2.)
  for i in range(pSz):
    if pY[i] == 0:
      for j in range(pSz):
        if i != j and pY[j] == 1:
          marg = - (B[j] - B[i]) / magW
          if marg < MARGB12[0]:
            MARGB12[0] = marg
            MARGB12[1] = B[i]
            MARGB12[2] = B[j]
            MARGB12[3] = i
            MARGB12[4] = j
  #draw minimal margin:
  pAx.annotate('Margin=' + str(MARGB12[0]), xy=(2., 2.3))
  pAx.annotate('Between #' + str(int(MARGB12[3])) + " and#" + str(int(MARGB12[4])), xy=(2., 1.4))
  return MARGB12
      

X = np.array([[.5, 1.2], [1., 1.], [3., 3.], [4., 3.3]])
Y = np.array([0, 0, 1, 1])
MRKS = ['^', 'o']
CLRS = ['green', 'blue']
fig, ax = plt.subplots(4)
smcnt = X.shape[0]

for i in range(smcnt):
  ax[0].scatter(X[i][0], X[i][1], marker=MRKS[Y[i]], c=CLRS[Y[i]])
  ax[1].scatter(X[i][0], X[i][1], marker=MRKS[Y[i]], c=CLRS[Y[i]])
  ax[2].scatter(X[i][0], X[i][1], marker=MRKS[Y[i]], c=CLRS[Y[i]])
  ax[3].scatter(X[i][0], X[i][1], marker=MRKS[Y[i]], c=CLRS[Y[i]])
W = np.array([1., 0.])
Wm = W.copy()
MARGB12m = prnHp(ax[0], X, Y, W, smcnt, 2.)
W[1] = 1.
MARGB12 = prnHp(ax[1], X, Y, W, smcnt, 2.)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy()
W[0] = 0.
MARGB12 = prnHp(ax[2], X, Y, W, smcnt, 2.)
if MARGB12m[0] < MARGB12[0]:
  MARGB12m = MARGB12
  Wm = W.copy()
b = MARGB12m[2] - ((MARGB12m[2] - MARGB12m[1]) / 2.)
i = int(MARGB12m[3])
x1a = X[i][0] - 2.
x1b = X[i][0] + 2.
x2a = -(x1a * Wm[0] / Wm[1]) - b
x2b = -(x1b * Wm[0] / Wm[1]) - b
ax[3].plot([x1a, x1b], [x2a, x2b], c="black")
ax[3].annotate('b=' + str(b), xy=(2., 2.3))

#ax[0].grid()
plt.show()
