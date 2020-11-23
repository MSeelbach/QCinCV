from dwave.system import DWaveSampler, EmbeddingComposite, VirtualGraphComposite

import neal
import numpy as np
from dimod.reference.samplers import ExactSolver
import dwave.inspector

import pickle



#sum of Pauli matrices equals constant. 
#Since variables were inserted we are in a lower dimension

def newPauli(position, N ):
   
    result= np.zeros(((N-1)**2+1))    

    result[(N-1)**2]=1
    if position==0:
        result[(N-1)**2]= 2-N
        for k in range((N-1)**2):
            result[k]+= 1
            
        return result
    

    if position <N:

       for k in range(N-1):
           result[position % N-1 + k *(N-1)]-= 1
       return result
    if position%N==0:

       for k in range(N-1):
           result[(position//(N)-1)*(N-1) +k] -= 1
       return result
    else:
        result[(N-1)**2]=0

        result[position- position//N - N]= 1
        return result
    return  result








   


   
def inserted(N,W,c):

    Wnew=np.zeros(((N-1)**2,(N-1)**2))
    cnew= np.zeros((N-1)**2)
    
    
    
    
    for i in range(N**2):
        for j in range(N**2):
            Wnew += W[i,j]* np.kron(newPauli(i, N)[0:(N-1)**2].reshape(1,(N-1)**2).T, newPauli(j, N )[0: (N-1)**2].reshape(1,(N-1)**2))
            cnew += W[i,j]* newPauli(i, N)[(N-1)**2] * newPauli(j, N )[0:(N-1)**2]
            cnew += W[i,j]* newPauli(i, N)[0:(N-1)**2]  * newPauli(j, N )[(N-1)**2]       
    
    
        cnew+= c[i,0]* newPauli(i,N)[0:(N-1)**2]
    
    
    
    
  
    
    
    optimizing=np.zeros((N-1)**2)
    
    for k in range((N-1)**2):
        optimizing[k]+= -np.abs(Wnew[k,k]) + np.abs(cnew[k]) #Minus because in the next line we get += 2W[k,k]
        for i in range((N-1)**2):
    
            optimizing[k]+=np.abs( Wnew[k,i]+ Wnew[i,k])
    
    
    
    MaxGrad= np.max(optimizing)
    
    # row and collumn-wise optimization
    Lambdaj=  np.zeros((2,N))
    
    
    
    
    columnSum= np.zeros(((N-1)**2,N-1))
    rowSum= np.zeros(((N-1)**2,N-1))
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
    
                if j==k:
                    columnSum[(N-1)*i+j,k]=  1        
                if i==k:
                    rowSum[(N-1)*i+j,k]=1
    
    for j in range(N-1):
        Lambdaj[0,j]= np.max(optimizing* columnSum[:,j])
        Lambdaj[1,j]= np.max(optimizing* rowSum[:,j])
    
    
    
    regularisationMatrix= np.zeros(((N-1)**2,(N-1)**2 ))
    regularisationVector= np.zeros(((N-1)**2,1 ))
    
    
    
    for i in range(N-1):
                regularisationMatrix +=( 1/2 *Lambdaj[0,i] + 1/2 *MaxGrad)* columnSum[:,i].reshape((N-1)**2,1)@  columnSum[:,i].reshape(1,(N-1)**2)
                regularisationVector += -( ( 1/2 * Lambdaj[0,i] + 1/2* MaxGrad) * columnSum[:, i ]).reshape(((N-1)**2,1))
                regularisationMatrix +=( 1/2 *Lambdaj[1,i] + 1/2 *MaxGrad)* rowSum[:,i].reshape((N-1)**2,1)@  rowSum[:,i].reshape(1,(N-1)**2)
                regularisationVector += -( ( 1/2 *Lambdaj[1,i] + 1/2* MaxGrad) * rowSum[:, i ]).reshape(((N-1)**2,1))
    
    
    regularisationMatrix+= MaxGrad/2 * np.ones(((N-1)**2,(N-1)**2))
    regularisationVector+= -MaxGrad/2 * (N-1+N-2 ) * np.ones(( (N-1)**2,1) )
    
    
    
    
    
    
    Wnew= Wnew+  regularisationMatrix
    cnew= cnew+  regularisationVector.T
    Qnew= Wnew/4 
    
    qunew= cnew.T/2 + (np.sum( Wnew, axis=0, keepdims= True ).T + np.sum(Wnew,axis= 1, keepdims=True) )/4
    bias=qunew.reshape((N-1)**2).tolist()
    
    J={}
    
    for i in range((N-1)**2):
    
        for j in range((N-1)**2):
    
            J.update( {(i,j): Qnew[i,j]})
    
    
    
  #  sampler = ExactSolver()
  #  response = sampler.sample_ising(bias,J)    
    
   # numberPrintedEnergies=0
  #  for datum in response.data(['sample', 'energy']): 
  #      if numberPrintedEnergies<2:    
       #     print(datum.sample, datum.energy)
     #   numberPrintedEnergies=numberPrintedEnergies+1
    
    
    solver = neal.SimulatedAnnealingSampler()
    response = solver.sample_ising(bias, J, num_reads=500)
    SimulatedResult=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences']):   
              #  print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                SimulatedResult.append([datum.sample,  datum.energy,  datum.num_occurrences]) 


   
    chain = np.max (bias)


    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_ising(bias,J,chain_strength=chain ,num_reads=500, return_embedding=True, anneal_schedule=((0.0,0.0),(40.0,0.5),(140.0,0.5),(180.0,1.0)))
    dwave.inspector.show(response)
    Result=[]
    for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
            print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
            Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])


    return [Result,response.info,SimulatedResult]
