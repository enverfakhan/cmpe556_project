#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:08:45 2019

@author: enverfakhan
"""

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


                
def importGlove(path, n):
    stop= set(stopwords.words('english'))
    stois= {}; vecs= []
    with open(path, 'r') as fh:
        cnt= 0
        for line in fh.readlines():
            ls= line.split(' ')
            if ls[0] in stop:
                continue
            stois[ls[0]]= cnt
            vecs.append([float(x) for x in ls[1:]])
            cnt+=1
            if cnt >= n:
                break
    return stois, np.array(vecs)

def sparser(adjMatrix, tresh):
    for i in range(len(adjMatrix)):
        for j in range(len(adjMatrix)):
            adjMatrix[i, j]= 0 if adjMatrix[i,j] < tresh else adjMatrix[i,j]


def argMax(modularity):
    maxim= 0
    for i in range(len(modularity)-1):
        for j in range(i+1, len(modularity)):
            val= modularity[i][j]
            if val > maxim:
                maxim= val
                pos= (i,j)
    
    if maxim >0:
        return pos
    else:
        return None
        
def createModularity(adjMatrix):
    
    degree= []; modularity= [list(v) for v in np.zeros_like(adjMatrix)]
    
    for row in adjMatrix:
        degree.append(sum(row))
    m= sum(degree)
    for i in range(len(adjMatrix)-1):
#        modularity.append([])
        for j in range(i+1, len(adjMatrix)):
            mod= (adjMatrix[i, j] - degree[i]*degree[j]/(2*m))/(2*m)
            modularity[i][j]= mod
            
    return modularity

def updateModularity(modularity, tup):
    i,j= tup[0], tup[1]
    vec= list(np.add(modularity[i], modularity[j]))
    vec[i]= (vec[i]+ vec[j])/2; del vec[j]
    del modularity[j]; 
    for v in modularity:
        del v[j]
    modularity[i]= vec
    for e in range(i):
        modularity[e][i]= vec[e]

def optimizer(adjMatrix):
    
    modularity= createModularity(adjMatrix)
    pos= argMax(modularity); assignments= []
    while  pos is not None:
        updateModularity(modularity, pos)
        assignments.append(pos)
        pos= argMax(modularity)
    
    return assignments

def findGroups(assignments, itos):
    a= [set([i]) for i in range(len(itos))]
    for tup in assignments:
        a[tup[0]].update(a[tup[1]])
        del a[tup[1]]
    b= [[itos[i] for i in c] for c in a]
    return b

if __name__ == '__main__':
    stois, vecs= importGlove('glove_vecctors.txt', 100)
    adjMatrix= cosine_similarity(vecs)
    del vecs
    sparser(adjMatrix, 0.7)
    assignments= optimizer(adjMatrix)
    itos= list(stois.keys())
    groups= findGroups(assignments, itos)
             
    