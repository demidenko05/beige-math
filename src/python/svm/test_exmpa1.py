#!/usr/bin/env python
# coding=UTF-8

import numpy as np      

#solve example 1 article 1 by linear equation (according to MIT MIT6_034F10_tutor05.pdf page 6 (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)):
#author Yury Demidenko

X = np.array ([[0.5, 1.2], [1.0, 1.0], [3.0, 3.0], [4.0, 3.3]])
Y = np.array ([-1, -1, 1, 1])
#System equation must consist of points that belongs to two support vectors - negative and positive, otherwise it's unsolvable, so make it for X_1 and X_2 (index from 0):
#NumPy linalg solver left part:
LP = np.array ([#constraint 1 sum (a_i*y_i) +0*b = 0:
                [Y[1],  Y[2], 0.0],
                #constraint 3 sum (a_i*y_i*X_i*X_1) + 1*b = -1
                [Y[1]*np.dot(X[1],X[1]),Y[2]*np.dot(X[2],X[1]), 1.0],
                #constraint 2 sum (a_i*y_i*X_i*X_2) + 1*b = 1
                [Y[1]*np.dot(X[1],X[2]),Y[2]*np.dot(X[2],X[2]), 1.0],
              ])
#NumPy linalg solver right part:
RP = np.array ([0.0, #constraint 1 sum (a_i*y_i) +0*b = 0
               Y[1], #constraint 3 sum (a_i*y_i*X_i*X_1) + 1*b = -1
               Y[2], #constraint 2 sum (a_i*y_i*X_i*X_2) + 1*b = +1
                ])
print ('Example 1 LP:\n', LP)
print ('Example 1 RP:\n', RP)
ALB = np.linalg.solve (LP, RP)
print ('Example 1 a0, a1, b:\n', ALB)

if not ( np.allclose (np.dot (LP, ALB), RP) ):
  print ('No met: np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

#W = sum(yi*ai*Xi)
W = Y[1]*ALB[0]*X[1] + Y[2]*ALB[0]*X[2]
print ('W:\n', W)
wMag = np.linalg.norm (W)
margStd = 2.0/wMag
print ('margin=2/|W|: ', margStd)

#NumPy linalg solver left part:
LP = np.array ([#constraint 1 sum (a_i*y_i) +0*b = 0:
                [Y[0], Y[1],  Y[2],  Y[3], 0.0],
                #constraint 3 sum (a_i*y_i*X_i*X_0) + 1*b = -1
                [Y[0]*np.dot(X[0],X[0]),Y[1]*np.dot(X[1],X[0]),Y[2]*np.dot(X[2],X[0]),Y[3]*np.dot(X[3],X[0]), 1.0],
                #constraint 3 sum (a_i*y_i*X_i*X_1) + 1*b = -1
                [Y[0]*np.dot(X[0],X[1]),Y[1]*np.dot(X[1],X[1]),Y[2]*np.dot(X[2],X[1]),Y[3]*np.dot(X[3],X[1]), 1.0],
                #constraint 2 sum (a_i*y_i*X_i*X_2) + 1*b = 1
                [Y[0]*np.dot(X[0],X[2]),Y[1]*np.dot(X[1],X[2]),Y[2]*np.dot(X[2],X[2]),Y[3]*np.dot(X[3],X[2]), 1.0],
                #constraint 2 sum (a_i*y_i*X_i*X_3) + 1*b = 1
                [Y[0]*np.dot(X[0],X[3]),Y[1]*np.dot(X[1],X[3]),Y[2]*np.dot(X[2],X[3]),Y[3]*np.dot(X[3],X[3]), 1.0],
              ])
#NumPy linalg solver right part:
RP = np.array ([0.0, #constraint 1 sum (a_i*y_i) +0*b = 0
               Y[0], #constraint 3 sum (a_i*y_i*X_i*X_0) + 1*b = -1
               Y[1], #constraint 3 sum (a_i*y_i*X_i*X_1) + 1*b = -1
               Y[2], #constraint 2 sum (a_i*y_i*X_i*X_2) + 1*b = +1
               Y[3]  #constraint 2 sum (a_i*y_i*X_i*X_3) + 1*b = +1
                ])
print ('Non-solvable LP:\n', LP)
print ('Non-solvable RP:\n', RP)
ALB = np.linalg.solve (LP, RP)
print ('Non-solvable  a0, a1, a3, a4, b:\n', ALB)

if np.allclose (np.dot (LP, ALB), RP):
  print ('Non-solvable No met: NOT np.allclose (np.dot (LP, ALB), RP)')
  exit (1)

print ('Test OK')
