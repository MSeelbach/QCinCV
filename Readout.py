import numpy as np

import scipy




import pickle

import scipy.io


#Use for N=4 (Inserted method gets recognized by having size like N=3)


def BinaryToDecimal(S,length):
    result=0
    for i in range(length):
        if S[i]== 1:
            result= result+ 2**(length -i-1)
    return result 

def application(svector,Matrix,Vektor):
    return  svector.T @Matrix@ svector + Vektor@svector

def isPermutation(number, length ):
    answer=True
    asVector= IntToVector(number,length )
    squareRoot=int(length**0.5)
    
    for k in range(squareRoot):
        
        sumHor=0
        sumVert=0
        
        for j in range( squareRoot):    
            sumHor= sumHor+ asVector[k+j*squareRoot]    
            sumVert= sumVert+ asVector[j+k* squareRoot ]
            
        if sumVert!=1:
            answer= False        
    
        if sumHor!= 1:
            answer=False
    return answer

def CouldBePermutation(number, length):
    answer=True    
    squareRoot=int(length**0.5)
    asVector= IntToVector(number, (squareRoot-1)**2)

    result= length* [0]

    for k in range (squareRoot-1):
        for j in range(squareRoot-1):
            result[k+1 +(squareRoot)*(j+1) ]= asVector[k +(squareRoot-1)*j]


    SumForZeroH=0
    SumForZeroV=0
    for k in range(1,squareRoot):
        SumH=0
        SumV=0
        for j in range(squareRoot-1): 
            SumV= SumV+ asVector[k-1 +(squareRoot-1)*(j)]
            SumH= SumH+ asVector[j +(squareRoot-1)*(k-1)]
        if SumH!=0 and SumH!=1:
            answer=False
    
        if SumV!=0 and SumV!=1:
            answer=False
        
            
        result[k]= 1- SumV
        result[(k)*(squareRoot)]=1- SumH
        
        SumForZeroH+= 1-SumH
        SumForZeroV+= 1-SumV
    
    if SumForZeroH!= SumForZeroV:
        answer=False
    if SumForZeroH!=0 and SumForZeroH!=1:
        answer=False
    
    result[0]= 1-SumForZeroH
    
    
    return [np.array(result) , answer]
    



    


def IntToVector( number ,length ):
    
    result= length*[0]
    restnumber= number
    for i in range(length):
        if restnumber% 2 ==1 :
            result[length-1-i]=1
            restnumber= (restnumber -1) 
        
        restnumber = restnumber //2
        
    return np.array( result)



#pickle.dump([Qneu,quneu], open( "save.p", "wb" ))
color = pickle.load( open( "stateHist.p", "rb" ) )

Instances=10
chainResults=[]

Wmatrices=[]
cvectors=[]

for chain in range(Instances):


    ReadOutAll= color[chain]
    
    W= ReadOutAll[0]
    c= ReadOutAll[1]
    
    Wmatrices.append(W)
    cvectors.append(c)
    
    
    N= W.shape[0]
    
    
    
    
    

    ResultList=[]
    i=0 
    for  element in ReadOutAll:
      

    
        if i >1: 
               ResultList.append(element)
        i= i+1
 
    #Handling of one list
    Results= []
    
    
    
    for Liste in ResultList:
    
        
            Information=np.zeros( (len( Liste),5) )
            
            
            for j in range(len(Liste)):
              
                if isinstance(Liste[j][0], list)==False:
                    
                    N= len(Liste[j][0])
                    
                    Information[j,0]= BinaryToDecimal( Liste[j][0],N)
                    Information[j,1]= Liste[j][1]
                    Information[j,2]= Liste[j][2]
                    if N==16:
                        Information[j,3]=application(  IntToVector(  Information[j,0],N), W,c.T) 
                    if N==9:
                        Information[j,3]=application(  CouldBePermutation(  Information[j,0],16)[0], W,c.T) 
                    Information[j,4]= Liste[j][3]
            Results.append(Information)
    chainResults.append(Results)
    
    
    
i=-1
EndResult=[]

for element in chainResults:
    i=i+1
    
    
    W= Wmatrices[i]
    c= cvectors[i]
    N= int(W.shape[0]**0.5)
    
    
    

         
         

    firstPermutationEvaluated=False    
  
    for j in range(2**(N**2)):
        if isPermutation(j,N**2):
             if firstPermutationEvaluated==False:
                bestPermutation=application(  IntToVector(j,N**2), W,c.T)
                worstPermutation=application(  IntToVector(j,N**2), W,c.T) 
                firstPermutationEvaluated=True
 
             candidate= application(  IntToVector(j,N**2), W,c.T)
         
             if candidate<bestPermutation:
                 bestPermutation= candidate
    for j in range(2**(N**2)):
        if isPermutation(j,N**2):
         
             candidate= application(  IntToVector(j,N**2), W,c.T)
         
             if candidate>worstPermutation:
                 worstPermutation= candidate
    
    
         
         
    Results= chainResults[i]

    
    
    
    
    


    shots=0
    for k in range(Results[0].shape[0]):
        shots+=Results[0][k,2] 
        
    
    Averages= np.zeros((6,len(Results)))
    
    for j in range(len(Results)):
    
        AverageWc=0
        MostProbable=[0,0]
        for k in range(Results[j].shape[0]):
            
          #  Results[j][k,3]=application( IntToVector( Results[j][k,0],9), W,c.T) 
          #  Results[j][k,4]= Results[j][k,3] - Results[j][k,1]
            if j >0 :
                if isPermutation(Results[j][k][0],N**2 ):
      
                    
                    AverageWc+=  Results[j][k,3] * Results[j][k,2]  /shots
                    
                    if Results[j][k,2]  /shots > MostProbable[0]:
                        MostProbable[0]= Results[j][k,2]  /shots
                        MostProbable[1]= Results[j][k,3]
                else:
                    AverageWc+=  worstPermutation * Results[j][k,2]  /shots
                    
                   # if Results[j][k,2]  /shots > MostProbable[0]:
                    #    MostProbable[0]= Results[j][k,2]  /shots
                     #   MostProbable[1]= worstPermutation
                    pass
                    
                        
            if j==0:
                if CouldBePermutation(  Results[j][k][0],16)[1] ==True:
                        
                    AverageWc+=  Results[j][k,3] * Results[j][k,2]  /shots
                    
                    if Results[j][k,2]  /shots > MostProbable[0]:
                        MostProbable[0]= Results[j][k,2]  /shots
                        MostProbable[1]= Results[j][k,3]

                else:
                    AverageWc+=  worstPermutation * Results[j][k,2]  /shots
                    
                   # if Results[j][k,2]  /shots > MostProbable[0]:
                    #    MostProbable[0]= Results[j][k,2]  /shots
                     #   MostProbable[1]= worstPermutation
                    pass
            
        coincidesWithOptimum=1
        if Results[j][0,3]== bestPermutation:
            print(True)
            
        else:
            print(False)#Consider as error message. For the dataset, where this exact code was used this was allways true
            
            coincidesWithOptimum=0
        Averages[0,j]=AverageWc
        Averages[1,j]=MostProbable[0]
        Averages[2,j]= MostProbable[1]
        Averages[3,j]=bestPermutation
        Averages[4,j]=worstPermutation
        Averages[5,j]=Results[j][0,2]/shots * coincidesWithOptimum
    EndResult.append(Averages)



Mittel= np.zeros(Averages.shape)
for t in range(len(EndResult)):
    for k in range( EndResult[t].shape[0] ):
        for l in range( EndResult[t].shape[1] ):
            Mittel[k,l]+= EndResult[t][k,l]/len(EndResult)         





import matplotlib.pyplot as plt

#mat = scipy.io.loadmat('energies_NVier.mat') for comparisons with other classical methods


x = np.linspace(1,10 , 10)

list1=[]
list2=[]
list3=[]
list4=[]

i=-1
mittel=0
for entry in EndResult:
    i=i+1
 #   list1.append(entry[2,0]-entry[3,0])
 #   list2.append(entry[2,1]-entry[3,0])
 #   list3.append(entry[2,2]-entry[3,0])
 #   list4.append(mat['energ'][i]-entry[3,0])
 #   mittel+= mat['energ'][i]/len(EndResult)

    list1.append(entry[5,0])
    list2.append(entry[5,1])
    list3.append(entry[5,2])
    

plt.plot(x,list1, marker= 'o', ms=14, markerfacecolor="None",
         #markeredgecolor='red',
         markeredgewidth=1 , ls= "None", label='Inserted')

plt.plot(x, list2, marker= "x",markersize = 10 , ls= "None", label='Baseline')
plt.plot(x, list3, marker= "+",markersize = 10 , ls= "None", label='Row-wise')
#plt.plot(x, list4, marker= "." , markersize = 8, ls= "None", label=r'$DS^{\star}$')






plt.plot(np.linspace(0.5,10.5 , 10),1/24* np.ones(10),"--")
plt.margins(x=0)


plt.legend(loc="upper right")



plt.xlabel('Instance')



#plt.ylabel("Normalized Energy")
plt.ylabel("Probability of finding the global optimum")






#plt.savefig("NVierVerbessert.eps")
plt.show()
    
    
    






