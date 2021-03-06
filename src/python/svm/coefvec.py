#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

import numpy as np
import matplotlib.pyplot as plt

#{x1,x2}*{w1,w2} + b =0 -> x2=-x1w1/w2 - b then w1 and w2!=0, x2=-b then w1=0, x1=-b then w2=0
def prnHypl(pW, pLen,pC,pAxn,pB):
  if pW[0] == 0.:
    ax[pAxn].plot([-pLen, pLen], [0. - pB, 0. - pB], c=pC)
  elif pW[1] == 0.:
    ax[pAxn].plot([0. - pB, 0. -pB], [-pLen, pLen], c=pC)
  else:
    ax[pAxn].plot([-pLen, pLen], [pLen*pW[0]/pW[1] - pB, -pLen*pW[0]/pW[1] - pB], c=pC)

fig, ax = plt.subplots(2)
leng = 4.
b = 0.
W = np.array([1., 0.])
prnHypl(W, leng, 'black', 0, b)
ax[0].annotate('W1{1,0} black', xy=(.2, 3.2))
b = -4.
prnHypl(W, leng, 'black', 1, b)
ax[1].annotate('W1{1,0} -4 black', xy=(.3, -2.2))
b = 0.
W = np.array([1., 1.])
prnHypl(W, leng, "green", 0, 0.)
ax[0].annotate('W2{1,1} green', xy=(-2.2, 2.2))
b = -4.
prnHypl(W, leng, 'green', 1, b)
ax[1].annotate('W2{1,1} -4 green', xy=(-2.2, 2.2 - b))
b = 0.
W = np.array([0., 1.])
prnHypl(W, leng, "blue", 0, 0.)
ax[0].annotate('W3{0,1} blue', xy=(1.2, .1))
b = -4.
prnHypl(W, leng, 'blue', 1, b)
ax[1].annotate('W3{0,1} -4 blue', xy=(1., .1 - b))
b = 0.
W = np.array([-1., 1.])
prnHypl(W, leng, "yellow", 0, 0.)
ax[0].annotate('W4{-1,1} yellow', xy=(-2.2, -2.2))
b = -4.
prnHypl(W, leng, 'yellow', 1, b)
ax[1].annotate('W4{-1,1} -4 yellow', xy=(-2.2, -2.2 - b))
ax[0].set(title='b=0:')
ax[0].grid()
ax[1].set(title='b=-4:')
ax[1].grid()
plt.show()
