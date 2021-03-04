#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#Graphic lib for vectors

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches

#Vector Algebra for drawing a vector's end
#for given line AB(vector) and triangle length
#returns 2 triangle's points (left, right), 3d is the line end B
def bsVecTrng(A, B, l):
  V = np.array([B[0] - A[0], B[1] - A[1]])
  V90 = np.array([-V[1], V[0]]) #rotated 90gr counterclockwise
  magV90 = np.sqrt(V90[0]**2. + V90[1]**2.) #=V's magnitude the same
  #unit vectors:
  U90 = V90 / magV90
  U = V / magV90
  #triangle base vectors:
  TBS = (magV90 - l) * U #base start
  TBNl = (l/2.0) * U90 #left 90gr
  TBNr = -1. * TBNl #right 90gr
  #base triangle point on the line:
  tbX = A[0] + TBS[0]
  tbY = A[1] + TBS[1]
  #result 2 points (left, right), 3d is the line end B:
  TRNGlr = np.array([[0.,0.],[0.,0.]])
  TRNGlr[0][0] = tbX + TBNl[0]
  TRNGlr[0][1] = tbY + TBNl[1]
  TRNGlr[1][0] = tbX + TBNr[0]
  TRNGlr[1][1] = tbY + TBNr[1]
  return TRNGlr

#It makes triangle path from given 3 points - left,  right, end
#and color
#returns mpatches.PathPatch
def bsTrngPth(TL, TR, TE, pColor):
  Path = mpath.Path
  path_data = [
    (Path.MOVETO, (TL[0], TL[1])),
    (Path.LINETO, TE),
    (Path.LINETO, (TR[0], TR[1])),
    (Path.CLOSEPOLY, (TL[0], TL[1])),
    ]
  codes, verts = zip(*path_data)
  path = mpath.Path(verts, codes)
  patch = mpatches.PathPatch(path, color=pColor)
  return patch
