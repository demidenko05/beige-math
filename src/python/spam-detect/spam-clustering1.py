#!/usr/bin/env python
# coding=UTF-8

# BSD 2-Clause License
# Copyright (c) 2021, Yury Demidenko (Beigesoft™)
# All rights reserved.
# See the LICENSE in the root source folder

# This application tries to separate spam and ham messages by using clustering, i.e. classify unlabeled data

# based on scikit-learn examples

# based on href="https://github.com/ksdkamesh99/Spam-Classifier
# see that project to make sure that:
# phrase "morefrmmob" stays unseparated by all methods 
# sentence "ye gauti sehwag odi seri" becames "yes gauti sehwag odi series" by lemmatiziers (probably wrong fixing), although they did not fix "u know" to "you know"

#dataset - https://www.kaggle.com/uciml/sms-spam-collection-dataset/download

#pass spam.csv full path as argument

import sys
import csv
import re
from time import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

ip = 0
pth = 'spam.csv'
for arg in sys.argv:
  if ip == 1:
    pth = arg
  ip += 1

nltk.download ('punkt') #tokenizer
nltk.download ('stopwords')
set (stopwords.words ('english'))
prtStm = PorterStemmer ()

dtSet = []
lbSet = []
#making dataset during reading CSV is more efficient way (takes less memory)
hsColNms = True
with open (pth, newline='', encoding="iso-8859-1") as csvFl:
  csvRdr = csv.reader (csvFl, delimiter=',', quotechar='"')
  isFst = hsColNms
  for row in csvRdr:
    if isFst: #skip column names
      isFst = False
      continue
    snts = re.sub ('[^A-Za-z]', ' ', row[1])
    snts = snts.lower ()
    wrds = word_tokenize (snts)
    wrdsPs = [prtStm.stem (i) for i in wrds if i not in stopwords.words ('english')]
    snts = ' '.join (wrdsPs)
    dtSet.append (snts)
    if row[0] == 'ham':
      lbSet.append (0)
    else:
      lbSet.append (1)

fstCnt = 100
print ('first %d messages:' % fstCnt)
for i in range (fstCnt):
  print (dtSet[i])

numCls = np.unique (lbSet).shape[0]
t0 = time ()
vctrzr = CountVectorizer (max_features=10000)  
print ("Extracting features from the training dataset using ", vctrzr)
X = vctrzr.fit_transform (dtSet)
print ("done in %fs" % (time() - t0))
print ("n_samples: %d, n_features: %d\n" % X.shape)

km = KMeans (n_clusters=numCls)
print ("Clustering sparse data with %s" % km)
t0 = time ()
predLbs = km.fit_predict (X)
print ("done in %0.3fs\n" % (time () - t0))

#check results:
  #from scikit-learn source code:
print ("Homogeneity: %0.3f" % metrics.homogeneity_score (lbSet, km.labels_))
print ("Completeness: %0.3f" % metrics.completeness_score (lbSet, km.labels_))
print ("V-measure: %0.3f" % metrics.v_measure_score (lbSet, km.labels_))
print ("Adjusted Rand-Index: %.3f"
       % metrics.adjusted_rand_score (lbSet, km.labels_))
print ("Silhouette Coefficient: %0.3f\n"
       % metrics.silhouette_score (X, km.labels_, sample_size=1000))

trmCnt = 20
print ("Top %d terms per cluster:" % trmCnt)
order_centroids = km.cluster_centers_.argsort ()[:, ::-1]
terms = vctrzr.get_feature_names ()

for i in range (numCls):
  print (" Cluster %d:" % i, end='')
  for ind in order_centroids[i, :trmCnt]:
    print (' %s' % terms[ind], end='')
  print ()

  #standard checking:
print ('First %d labels source-cluster:' % fstCnt)
#kmlbs = km.labels_
kmlbs = predLbs
for i in range (fstCnt):
  print (lbSet[i], '-', kmlbs[i], end='; ')
wrng = 0;
wrngSpm = 0;
totSpm = 0;
tot = X.shape[0]
for i in range (tot):
  if lbSet[i] == 1:
    totSpm += 1
    if lbSet[i] != kmlbs[i]:
      wrngSpm += 1
  if lbSet[i] != kmlbs[i]:
    wrng += 1
#TODO depending of previous KMEAN's labels results wrng inverses (e.g Homogeneity > 0.3?)!
accur = (tot - wrng) / tot * 100.0
accurSpm = (totSpm - wrngSpm) / totSpm * 100.0
print ('\nAccuracy total = ' + str (accur) + '%')
print ('\nSpam total = ' + str (totSpm) + ' Accuracy spam = ' + str (accurSpm) + '%')
