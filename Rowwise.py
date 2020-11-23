

# ------ Import necessary packages ----
from dwave.system import DWaveSampler, EmbeddingComposite,VirtualGraphComposite

import neal
import numpy as np
from dimod.reference.samplers import ExactSolver
import dwave.inspector




import pickle



def rowwise(N,W,c): 
    

      
    optimizing=np.zeros(N**2)
    
    for k in range(N**2):
        optimizing[k]+= -np.abs(W[k,k]) + np.abs(c[k]) #Minus because in the next line we get += 2W[k,k]
        for i in range(N**2):
        
            optimizing[k]+=np.abs( W[k,i]+ W[i,k])
    
    
    
    MaxGrad= np.max(optimizing)
    
    # column and row-wise optimization:
    Lambdaj=  np.zeros((2,N))
        
    
    
    qu= c/2 + (np.sum( W, axis=0, keepdims= True ).T + np.sum(W,axis= 1, keepdims=True) )/4
    columnSum= np.zeros((N**2,N))
    rowSum= np.zeros((N**2,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                
                if j==k:
                    columnSum[N*i+j,k]=  1        
                if i==k:
                    rowSum[N*i+j,k]=1
    
    for j in range(N):
        Lambdaj[0,j]= np.max(optimizing* columnSum[:,j])
        Lambdaj[1,j]= np.max(optimizing* rowSum[:,j])
      
      
        
    regularisationMatrix= np.zeros((N**2,N**2 ))
    regularisationVector= np.zeros((N**2,1 ))
    LambdaValues=10
    
    
    for i in range(N):
                regularisationMatrix +=( Lambdaj[0,i] + 1/2 *MaxGrad)* columnSum[:,i].reshape(N**2,1)@  columnSum[:,i].reshape(1,N**2)
                regularisationVector += -(2* ( Lambdaj[0,i] + 1/2* MaxGrad) * columnSum[:, i ]).reshape((N**2,1))
                regularisationMatrix +=( Lambdaj[1,i] + 1/2 *MaxGrad)* rowSum[:,i].reshape(N**2,1)@  rowSum[:,i].reshape(1,N**2)
                regularisationVector += -(2* ( Lambdaj[1,i] + 1/2* MaxGrad) * rowSum[:, i ]).reshape((N**2,1))
        
    
    
    
    
    

    
    
    
    
    
    W= W+ regularisationMatrix
    c= c+ regularisationVector
    Q= W/4 
    
    qu= c/2 + (np.sum( W, axis=0, keepdims= True ).T + np.sum(W,axis= 1, keepdims=True) )/4
    
    for i in range(0,N**2):
        Q[i,i]=0
    
    
        
    
    bias=qu.reshape(N**2).tolist()
    
    J={}
    
    for i in range(N**2):
    
        for j in range(N**2):
            
            J.update( {(i,j): Q[i,j]})
    
    
    
   # sampler = ExactSolver()
    #response = sampler.sample_ising(bias,J)    
    
   # numberPrintedEnergies=0
   # for datum in response.data(['sample', 'energy']): 
   #     if numberPrintedEnergies<3:    
    #        print(datum.sample, datum.energy)
    #    numberPrintedEnergies=numberPrintedEnergies+1
    
    
  #  solver = neal.SimulatedAnnealingSampler()
  #  response = solver.sample_ising(bias, J, num_reads=50)
  #  Result=[]
   # for datum in response.data(['sample', 'energy', 'num_occurrences']):   
   #             print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
   #             Result.append([datum.sample,  datum.energy,  datum.num_occurrences]) 

    
    
    
   # sampler = EmbeddingComposite(DWaveSampler(annealing_time=40))
    

    solver = neal.SimulatedAnnealingSampler()
    response = solver.sample_ising(bias, J, num_reads=500)
    SimulatedResult=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences']):   
                #print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                SimulatedResult.append([datum.sample,  datum.energy,  datum.num_occurrences]) 


    
    
   


    chain = np.max (bias)


    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_ising(bias,J,chain_strength=chain  ,num_reads=500, return_embedding=True,anneal_schedule=((0.0,0.0),(40.0,0.5),(140.0,0.5),(180.0,1.0)))

    dwave.inspector.show(response)
    Result=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
            print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
            Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])

    return [Result,response.info,SimulatedResult]
