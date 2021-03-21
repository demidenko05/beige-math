#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

#transfer NIST data to LIBSVM formatted train (900 samples) and test (the rest 100) files
#NIST data from http://www.cis.jhu.edu/~sachin/digit/digit.html 1000 28x28 digits (unsigned char 8bit): data0..data9
#required path to the data in command line sys.argv[1]

import sys, os
sys.path += [os.path.dirname(os.path.abspath (__file__)) + '/../..']
from BsLibMisc import *
import numpy as np

def prnUsage ():
  print ('You must pass path to the NIST files 1000 28x28 digits (unsigned char 8bit): data0..data9!')
  print ('from http://www.cis.jhu.edu/~sachin/digit/digit.html')
  print ('Use: python dig1000tolibsvm.py [path_to_nist_files]')

ip = 0
pth = ''
for arg in sys.argv:
  if ip == 1:
    pth = arg
  ip += 1

if pth == '' or pth == '-h':
  prnUsage ()
  exit (1)

digCnt = 1000
digSz = 28
smpCnt = digSz * digSz

trainCnt = 900
testCnt = digCnt - trainCnt
testOfst = trainCnt * smpCnt

ftrain = False
ftest = False

try:

  for dig in range (10):

    fnme = pth + "/data" + str (dig)
    NUMd = np.fromfile (fnme, dtype=np.uint8)

    if NUMd.shape[0] != digCnt * smpCnt:
      print ('It must be 1000 uint8 28x28 samples in ', fnme)
      raise
      
    #just for visual control, print several samples:
    print ("Samples #0,1,2,900,901,902 From file: ", fnme)
    bsPrnImgTxt (NUMd, 0, digSz)
    bsPrnImgTxt (NUMd, 1*smpCnt, digSz)
    bsPrnImgTxt (NUMd, 2*smpCnt, digSz)
    bsPrnImgTxt (NUMd, 900*smpCnt, digSz)
    bsPrnImgTxt (NUMd, 901*smpCnt, digSz)
    bsPrnImgTxt (NUMd, 902*smpCnt, digSz)
    #there is 5 #900 that looks like 6

    if ftrain == False:
      ftrain = open (pth + '/nist'+str(digCnt)+'x'+str(digSz)+'x'+str(digSz), 'w')

    for i in range (trainCnt):
      ftrain.write (str (dig))
      for j in range (smpCnt):
        ftrain.write(' ' + str (j+1) + ':' + str (NUMd[i * smpCnt + j]))
      ftrain.write('\n')

    if ftest == False:
      ftest = open (pth + '/nist'+str (digCnt)+'x'+str(digSz)+'x'+str(digSz) + 't', 'w')

    for i in range (testCnt):
      ftest.write (str (dig))
      for j in range (smpCnt):
        ftest.write(' ' + str (j+1) + ':' + str (NUMd[testOfst + i * smpCnt + j]))
      ftest.write('\n')

except:
  prnUsage ()
  print (sys.exc_info ()[0])
  if ftrain != False:
    ftrain.close()
  if ftest != False:
    ftest.close()
  raise

if ftrain != False:
  ftrain.close()

if ftest != False:
    ftest.close()
