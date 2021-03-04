#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import sys, os
sys.path += [os.path.dirname(os.path.abspath(__file__)) + '/..']
from BsLibGVec import *
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[.5, 1.2], [1., 1.], [3., 3.], [4., 3.3]])
Y = np.array([0, 0, 1, 1])
MRKS = ['^', 'o']
CLRS = ['green', 'blue']
fig, ax = plt.subplots()
for i in range(X.shape[0]):
  ax.scatter(X[i-1][0], X[i-1][1], marker=MRKS[Y[i-1]], c=CLRS[Y[i-1]])
#Hyperplane HP y=-x+4 (x2=-x1+4) -> x1 + x2 -4 = 0 -> W*X+b=0 -> [1,1]*X -4=0, W=[1,1] WX=0 is Hyperplane through origin, b is shifting along the last axis (X2 here)
bHP = 4.0
ax.plot([0., bHP], [bHP, 0.], c='black', linewidth=1)
W = np.array([1., 1.])
#point A on HP to draw W:
hpaX = 3.
HPA = np.array([hpaX, -hpaX + bHP])
Wa = np.array([HPA, [HPA[0] + W[0], HPA[1] + W[1]]])
ax.plot([Wa[0][0], Wa[1][0]], [Wa[0][1], Wa[1][1]], c='black', linewidth=0.5)
TRNGlr = bsVecTrng(Wa[0], Wa[1], 0.15)
patch = bsTrngPth(TRNGlr[0], TRNGlr[1], Wa[1], 'black')
ax.add_patch(patch)
ax.annotate('W', xy=(3.2, 1.4))
ax.plot([0,2], [2,0], c=CLRS[0], linewidth=1)
ax.plot([0,6], [6,0], c=CLRS[1], linewidth=1)
ax.annotate('H1', xy=(.13, 2.))
ax.annotate('H2', xy=(5.1, 1.))
ax.set(xlabel='x1', ylabel='x2', title='Simplest example in 2D: 2x2 points\nX1[0.5, 1.2], X2[1, 1] = class1(green)\nX3[3, 3], X4[4, 3.3] = class2(blue)\nX2[1, 1] and X3[3, 3] - belongs to the support vectors H1 and H2')
ax.grid()
plt.show()
