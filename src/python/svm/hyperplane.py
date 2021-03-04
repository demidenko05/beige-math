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

fig, ax = plt.subplots()
ax.plot([-3., 3.], [3., -3.], c='black', linewidth=1)
ax.plot([0., 4.], [4., 0.], c='black', linewidth=1)
ax.plot([0., 1.], [0., -1.], c='green', linewidth=2)
W = np.array([1., 1.])
#point A on HP to draw W:
hpaX = 1.
HPA = np.array([hpaX, -hpaX])
Wa = np.array([HPA, [HPA[0] + W[0], HPA[1] + W[1]]])
ax.plot([Wa[0][0], Wa[1][0]], [Wa[0][1], Wa[1][1]], c='black', linewidth=0.5)
TRNGlr = bsVecTrng(Wa[0], Wa[1], 0.15)
patch = bsTrngPth(TRNGlr[0], TRNGlr[1], Wa[1], 'black')
ax.add_patch(patch)
TRNGlr2 = bsVecTrng([0., 0], [1.,-1.], 0.20)
patch2 = bsTrngPth(TRNGlr2[0], TRNGlr2[1], [1., -1], 'green')
ax.add_patch(patch2)
ax.annotate('O(0,0)', xy=(0.1, 0.1))
ax.annotate('A(1,-1)', xy=(0.2, -1.2))
ax.annotate('OA{1,-1}', xy=(0.45, -0.4))
ax.annotate('W{1,1}', xy=(1.4, -0.9))
ax.annotate('HP {x1,x2}*{1,1}=0', xy=(-2.2, 2.4))
ax.annotate('HP shifted by 4', xy=(2.7, 1.4))
ax.set(xlabel='x1(or x)', ylabel='x2 (or y)', title='Hyperplane {x1,x2}*{1,1}=0, its shifted by 4 clone,\nnormal W{1,1} from A(1,-1)')
ax.grid()
plt.show()
