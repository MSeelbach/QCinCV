#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:02:48 2020

@author: cv_marcel
"""

import Inserted 
import Rowwise
import Baseline
import numpy as np

import pickle

import scipy.io
mat = scipy.io.loadmat('NVier.mat')



numberInstances= 10
N=4

Save=[]
Details=[]
Simulated=[]

#embeddingsSave = pickle.load( open( "saveDetails4.p", "rb" ) )
#print(embeddingsSave[2][1]['embedding_context']['embedding']  )

for k in range(numberInstances):
    
    W= np.random.uniform(-1,1,(N**2, N**2) )
    c= np.random.uniform(-1,1,( N**2,1) )

    #W= mat['W'][k]
    #c= mat['c'][k]

    
   

    one=Inserted.inserted(N,W, c)
    two=Baseline.baseline(N,W,c)
    three=Rowwise.rowwise(N, W, c)


    Save.append([W,c,one[0],two[0], three[0]])
    Details.append([one[1], two[1], three[1]])
    Simulated.append( [one[2], two[2], three[2]] )


    pickle.dump(Save, open( "StateHist.p", "wb" ))
    pickle.dump(Details, open("SolverDetails.p","wb"))
    pickle.dump(Simulated, open("SimulatedAnnealing.p", "wb"))

