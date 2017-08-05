#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:46:34 2017

@author: michaelxu
"""
import numpy
import gzip
import random
import math
#from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before
##################################################
# 5.Alpha                                        #
##################################################
allUser = []
allItem = []
allRatings = []
for l in readGz("train.json.gz"):
  allUser.append(l['reviewerID'])
  allItem.append(l['itemID'])
  allRatings.append(l['rating'])
num_train = 200000
trainUser = allUser[:num_train]
trainItem = allItem[:num_train]
trainRatings = allRatings[:num_train]

#Initilization
p=100
Ga1,Ga2=0.008,0.01
La1,La2=0.8,0.8
train_ave = sum(trainRatings)*1.0/len(trainRatings)
mse = sum([(x-train_ave)**2 for x in trainRatings])*1.0/len(trainRatings)
print(train_ave,mse)
#Rating/PreRating
iRate,uRate,PreRate,Err = dict(),dict(),dict(),dict()
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if item not in iRate:
        iRate[item] = dict()
    if user not in iRate[item]:
        iRate[item][user] = rate

    if user not in uRate:
        uRate[user] = dict()
    if item not in uRate[user]:
        uRate[user][item] = rate

    if user not in PreRate:
        PreRate[user] = dict()
    if item not in PreRate[user]:
        PreRate[user][item] = random.gauss(train_ave,0.5)
        #PreRate[user][item] = 5*random.random()

    if user not in Err:
        Err[user] = dict()
    if item not in Err:
        Err[user][item] = random.gauss(0,0.01)
#mu mean ratings
mu = sum(trainRatings)*1.0/len(trainRatings)
#Bu Beta of User/Bi Beta of Item
Bu,Bi = dict(),dict()
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if item not in Bi:
        Bi[item] = 0.0
        for user in iRate[item]:
            Bi[item] += iRate[item][user]-mu
        Bi[item]/= 20.0+len(iRate[item])
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if user not in Bu:
        Bu[user] = 0.0
        for item in uRate[user]:
            Bu[user] += uRate[user][item]-mu-Bi[item]
        Bu[user]/= 8.0+len(uRate[user])        
#pu/qi Latent Factor 
pu,qi = dict(),dict()
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if user not in pu:
        pu[user] = [random.gauss(0,0.1)/math.sqrt(p) for x in range(p)]
    if item not in qi:
        qi[item] = [random.gauss(0,0.1)/math.sqrt(p) for x in range(p)]
#yj Feedback
#yj = dict()
#for l in range(len(trainUser)):
#    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
#    if user not in yj:
#        yj[user] = dict()
#for u in yj:
#    for l in range(len(trainUser)):
#        user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
#        if item not in yj[user]:
#            yj[user][item]=[0.0 for x in range(p)]
#yj = dict()
#for l in range(len(trainUser)):
#    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
#    if user not in yj:
#        yj[user] = dict()
#    if item not in yj[user]:
#        yj[user][item]=[random.gauss(0,0.01) for x in range(p)]
yj = dict()
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if item not in yj:
        yj[item] = [random.gauss(0,0.1)/math.sqrt(p) for x in range(p)]
#Nu Number of User related
Nu = dict()
for l in range(len(trainUser)):
    user,item,rate = trainUser[l],trainItem[l],trainRatings[l]
    if user not in Nu:
        Nu[user] = 0
    else:
        Nu[user] += 1
for u in Nu:
    if Nu[u]>1:
        Nu[u] = 1/math.sqrt(Nu[u])
    else:
        Nu[u] = 0#？
num_test =10000
testUser = allUser[-num_test:]
testItem = allItem[-num_test:]
test_Ratings = allRatings[-num_test:]
testRatings = dict()
for i in range(num_test):
    user,item,value = testUser[i],testItem[i],test_Ratings[i]
    if user not in testRatings:
        testRatings[user] = dict()
    if item not in testRatings[user]:
        testRatings[user][item]=value
testPreRatings = dict()
for i in range(num_test):
    user,item,value = testUser[i],testItem[i],test_Ratings[i]
    if user not in testPreRatings:
        testPreRatings[user] = dict()
    if item not in testPreRatings[user]:
        testPreRatings[user][item]=value
print('Start')
def score(x):
    if x>5:
        x=5
    if x<1:
        x=1
    return x
################################################
# Training
################################################
E = 100
for index in range(50):
    Ga1 *= 0.85 + 0.1*random.random()
    Ga2 *= 0.85 + 0.1*random.random()
    for u in Err:
        for i in Err[u]:
            Err[u][i] = uRate[u][i]-PreRate[u][i]
    for u in uRate:
        for i in uRate[u]:
            Bu[u] = Bu[u]+Ga1*(Err[u][i]-La1*Bu[u])
    for i in iRate:
        for u in iRate[i]:
            Bi[i] = Bi[i]+Ga1*(Err[u][i]-La1*Bi[i])
    for i in iRate:
        for u in iRate[i]:
            qi[i] = qi[i]+Ga2*(Err[u][i]*(pu[u]+Nu[u]*numpy.sum([yj[j] for j in uRate[u] ],axis=0))-numpy.dot(La2,qi[i]))
    for u in uRate:
        for i in uRate[u]:
            pu[u] = pu[u]+Ga2*(Err[u][i]*qi[i]-numpy.dot(La2,pu[u]))
    for u in uRate:
        for j in uRate[u]:
            for i in uRate[u]:
                yj[j] = yj[j]+Ga2*(Err[u][i]*numpy.dot(Nu[u],qi[i])-numpy.dot(La2,yj[j]))
    for u in PreRate:
        for i in PreRate[u]:
            PreRate[u][i] = mu+Bu[u]+Bi[i]+numpy.dot(numpy.transpose(qi[i]),(pu[u]+Nu[u]*numpy.sum([yj[j] for j in uRate[u]],axis=0)))
    Error = 0.0
    for u in Err:
        for i in Err[u]:
            Error += Err[u][i]**2
    for u in testPreRatings:
        for i in testPreRatings[u]:
            if u in Bu and i in Bi:
                testPreRatings[u][i] = mu+Bu[u]+Bi[i]+numpy.dot(numpy.transpose(qi[i]),(pu[u]+Nu[u]*numpy.sum([yj[j] for j in uRate[u]],axis=0)))
                testPreRatings[u][i] = score(testPreRatings[u][i])
            if u in Bu and i not in Bi:
                testPreRatings[u][i] = mu+Bu[u]
                testPreRatings[u][i] = score(testPreRatings[u][i])
            if u not in Bu and i in Bi:
                testPreRatings[u][i] = mu+Bi[i]
                testPreRatings[u][i] = score(testPreRatings[u][i])
            if u not in Bu and i not in Bi:
                testPreRatings[u][i] = mu
                testPreRatings[u][i] = score(testPreRatings[u][i])
    Error1 = 0.0
    for u in testPreRatings:
        for i in testPreRatings[u]:
            Error1 += (testPreRatings[u][i]-testRatings[u][i])**2
    print(index,Error/num_train,Error1/num_test)
    if E-Error1/10000 < 0.00001:
        break
    else:
        E = Error1
#########################################################
##8.Kaggle                                              #
#########################################################
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if u in Bu and i in Bi:
      predictions.write(u + '-' + i + ',' + str(score(mu+Bu[u]+Bi[i]+numpy.dot(numpy.transpose(qi[i]),(pu[u]+Nu[u]*numpy.sum([yj[j] for j in uRate[u]],axis=0))))) + '\n')
  if u in Bu and i not in Bi:
      predictions.write(u + '-' + i + ',' + str(score(mu+Bu[u])) + '\n')
  if u not in Bu and i in Bi:
      predictions.write(u + '-' + i + ',' + str(score(mu+Bi[i])) + '\n')
  if u not in Bu and i not in Bi:
      predictions.write(u + '-' + i + ',' + str(score(mu)) + '\n')
predictions.close()