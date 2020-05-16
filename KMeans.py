from dwave.system import DWaveSampler, EmbeddingComposite

import matplotlib.pyplot as plt
import numpy as np

#Algorithm for K-Means-Clustering:

#@inproceedings{bauckhage2017ising,
#  title={Ising Models for Binary Clustering via Adiabatic Quantum Computing},
#  author={Bauckhage, Christian and Brito, Eduardo and Cvejoski, Kostadin and Ojeda, C{\'e}sar and Sifa, Rafet and Wrobel, Stefan},
#  booktitle={International Workshop on Energy Minimization Methods in Computer Vision and Pattern Recognition},
#  pages={3--17},
#  year={2017},
#  organization={Springer}
#}

N=15

X= np.random.rand(2,N)
mean= np.sum(X, axis= 1 )/N
X= X - np.ones((2,N))*mean.reshape((2,1))
Q= (X.T)@X


J={}

bias= N*[0.]

for i in range(N):

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

plotResult(DictToList(mostProbable, N),0)