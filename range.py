from dwave.system import DWaveSampler, EmbeddingComposite

import matplotlib.pyplot as plt
import numpy as np


N=15

X= np.random.rand(2,N)
mean= np.sum(X, axis= 1 )/N
X= X - np.ones((2,N))*mean.reshape((2,1))
Q= (X.T)@X





#Equality restriction \sum_i s_i= c


c=N-2*3 


constLambda= np.sum(np.sum( np.absolute(Q),0) ,1)
#Is the problem itself solved or does one just generate states that fullfill the constraints?


ratio= np.max(Q)


Q+= constLambda* np.ones((N,N))
qu+= 2 * constLambda* c *np.ones(N)

ratio = ratio/constLambda

print(ratio)

J={}

bias=[]

for i in range(N):
    bias.append(qu[i] )
    for j in range(N):
        
        J.update( {(i,j): Q[i,j]})

sampler = EmbeddingComposite(DWaveSampler())


response = sampler.sample_ising(bias,J, num_reads=50)

ResultList=[]

for datum in response.data(['sample', 'energy', 'num_occurrences']):   
    print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
    ResultList.append([datum.sample, datum.num_occurrences])



Occurrence=0

for element in  ResultList:
    if element[1]>Occurrence:
        Occurrence= element[1]
        mostProbable= element[0]


def plotResult(ClusterZugehoerigkeit,inFigure ):
    plt.figure(inFigure)
    plt.scatter(X[0,:], X[1,:],c= ClusterZugehoerigkeit)

def DictToList( dict , N):
    result=[]
    for k in range(N):
        if dict[k]== -1:
            result.append(1)
        if dict[k]==1:
            result.append(0)
    return result
print(mostProbable)
print(DictToList(mostProbable, N))
