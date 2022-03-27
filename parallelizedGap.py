import itertools
import multiprocessing as mp
import numpy as np
import os, pickle
from tqdm import tqdm, trange
import math
import numpy.linalg as npl
import itertools
import scipy.special,time, scipy.stats
from collections import Counter
from scipy.special import comb as choose
import sklearn.cluster
from sklearn.metrics import pairwise_distances as dist



def compGap(data,k):
    
    if k==1:
#         dists = sklearn.metrics.pairwise_distances(data,metric='sqeuclidean')
        dists = dist(data,metric='sqeuclidean')
        return dists.sum()/4/data.shape[0]
    
    centroid,labels,inertia = sklearn.cluster.k_means(data,k)
    D = np.zeros(k)
    counts = np.zeros(k)
    for (i,li) in enumerate(labels):
        counts[li]+=1
        for (j,lj) in enumerate(labels):
            if li != lj or j <= i:
                continue
            D[li] += np.linalg.norm(data[i]-data[j])**2
    
    return np.sum(D/counts/2)
    
    

def gapStatBrute(dataSet,kMax=20,maxPulls=20,seed=0):
    np.random.seed(seed)
    lows = np.min(dataSet,axis=0)
    highs = np.max(dataSet,axis=0)
    n,d=dataSet.shape
    
    B=maxPulls
    dataFakes = np.zeros((B,n,d))
    for t in range(B):
        dataFake = np.zeros((n,d))
        for i in range(d):
            dataFake[:,i] = np.random.uniform(lows[i],highs[i],size=n)
        dataFakes[t] = dataFake
    

    WkArr = np.zeros(kMax)
    WkbArr = np.zeros((kMax,B))
    
    for k in range(1,kMax):

        WkArr[k] = compGap(dataSet,k)


        for t in range(B):

            dataFake = dataFakes[t]
            
            WkbArr[k,t] = compGap(dataFake,k)



        if k>1 and np.mean(WkbArr[k-1]) - WkArr[k-1]>= np.mean(WkbArr[k]) - WkArr[k] -np.sqrt(1+1/B)*np.std(WkbArr[k]):
            return k-1,k*B,WkArr,WkbArr

        
    return kMax,kMax*B,WkArr,WkbArr

    

def gapStat(dataSet,kMax=20,maxPulls=20,const=5,seed=0):
    np.random.seed(seed)
    lows = np.min(dataSet,axis=0)
    highs = np.max(dataSet,axis=0)
    n,d=dataSet.shape
    
    B=maxPulls
    dataFakes = np.zeros((B,n,d))
    for t in range(B):
        dataFake = np.zeros((n,d))
        for i in range(d):
            dataFake[:,i] = np.random.uniform(lows[i],highs[i],size=n)
        dataFakes[t] = dataFake
    

    WkArr = np.zeros(kMax)
    WkbArr = np.zeros((kMax,maxPulls))
    numPulls = np.zeros(kMax,dtype='int')

#     prevArr = []
#     currArr = []

    WkArr[1] = compGap(dataSet,1)
    
    numInitPulls = 1 #### used 3 for real sims
    for t in range(numInitPulls):
        WkbArr[1,t] = compGap(dataFakes[t],1)
    numPulls[1]=numInitPulls
    
    k=2
    while k<kMax:
        currArr = []
        WkArr[k] = compGap(dataSet,k)
        
        for t in range(numInitPulls):
            WkbArr[k,t] = compGap(dataFakes[t],k)
        numPulls[k]=numInitPulls
            
        
        while numPulls[k-1]<maxPulls:
            WkbArr[k-1,numPulls[k-1]] = compGap(dataFakes[numPulls[k-1]],k-1)
            WkbArr[k,numPulls[k]] = compGap(dataFakes[numPulls[k]],k)

            numPulls[k]+=1
            numPulls[k-1]+=1
            
            
            currArr = WkbArr[k,:numPulls[k]]
            prevArr = WkbArr[k-1,:numPulls[k-1]]
            if np.mean(prevArr) - WkArr[k-1] - const*np.std(prevArr)/np.sqrt(len(prevArr)) >= np.mean(currArr) - WkArr[k] +(const/np.sqrt(len(currArr)) -1)*np.std(currArr):
                return k-1,numPulls.sum(),WkArr,WkbArr,numPulls

            if np.mean(prevArr) - WkArr[k-1] + const*np.std(prevArr)/np.sqrt(len(prevArr)) <= np.mean(currArr) - WkArr[k] -(const/np.sqrt(len(currArr)) +1)*np.std(currArr):
                break


            
                
            t+=1

        if numPulls[k-1] == maxPulls and np.mean(prevArr) - WkArr[k-1] >= np.mean(currArr) - WkArr[k] -(1+1.0/np.sqrt(numPulls[k]))*np.std(currArr):
            return k-1,numPulls.sum(),WkArr,WkbArr,numPulls
            
        k+=1
        
    return kMax,numPulls.sum(),WkArr,WkbArr,numPulls


# def createDataset(seed): ### i1
#     np.random.seed(seed)
#     return np.random.uniform(size=(200,10))

# def createDataset(seed): ###i2
#     np.random.seed(seed)
    
#     dataSet = np.random.normal(size=(100,2),scale=.5)
#     dataSet[25:50] += [0,5]
#     dataSet[50:] += [5,-3]
        
#     return dataSet


# def createDataset(seed): ###i3
#     np.random.seed(seed)
    
#     clusterSizes = np.random.choice([25,50],size=4)
#     n = np.sum(clusterSizes)
    
#     d=3
#     numClusters=4
    
#     while True:
#         clusterCenters = np.random.normal(size=(numClusters,d),scale=4)
#         valid = True
#         for i in range(numClusters):
#             for j in range(i+1,numClusters):
#                 if np.linalg.norm(clusterCenters[i]-clusterCenters[j]) <=4:
#                     valid = False
#         if valid:
#             break
        
#     dataSet = np.random.normal(size=(n,d),scale=.1)
#     lens = np.cumsum(clusterSizes)
#     dataSet[:lens[0]] +=clusterCenters[0]
#     for i in range(numClusters-1):
#         dataSet[lens[i]:lens[i+1]] += clusterCenters[i+1]
#     return dataSet



def createDataset(seed): ###i4
    np.random.seed(seed)
    
    clusterSizes = np.random.choice([25,50],size=4)
    n = np.sum(clusterSizes)
    
    d=10
    numClusters=4
    
    while True:
        clusterCenters = np.random.normal(size=(numClusters,d),scale=5) ## real scale was 4
        valid = True
        for i in range(numClusters):
            for j in range(i+1,numClusters):
                if np.linalg.norm(clusterCenters[i]-clusterCenters[j]) <=1:
                    valid = False
        if valid:
            break
        
    dataSet = np.random.normal(size=(n,d),scale=1)
    lens = np.cumsum(clusterSizes)
    dataSet[:lens[0]] +=clusterCenters[0]
    for i in range(numClusters-1):
        dataSet[lens[i]:lens[i+1]] += clusterCenters[i+1]
    return dataSet


def createDataset(seed): ###i5
    np.random.seed(seed)
    
    numClusters=10
    clusterSizes = np.random.choice([25,50],size=numClusters)
    n = np.sum(clusterSizes)
    
    d=10
    
        
    
    while True:
        clusterCenters = np.random.normal(size=(numClusters,d))
        clusterCenters= clusterCenters / np.linalg.norm(clusterCenters,axis=1,keepdims=True)*5
        valid = True
        for i in range(numClusters):
            for j in range(i+1,numClusters):
                if np.linalg.norm(clusterCenters[i]-clusterCenters[j]) <=5:
                    valid = False
        if valid:
            break
        
    dataSet = np.random.normal(size=(n,d),scale=.3)
    lens = np.cumsum(clusterSizes)
    dataSet[:lens[0]] +=clusterCenters[0]
    for i in range(numClusters-1):
        dataSet[lens[i]:lens[i+1]] += clusterCenters[i+1]
    return dataSet



def createDataset(seed): ###i5
    np.random.seed(seed)
    
    numClusters=4
    clusterSizes = np.random.choice([25,50],size=numClusters)
    n = np.sum(clusterSizes)
    
    d=2
    
        
    
    clusterCenters = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        
    dataSet = np.random.normal(size=(n,d),scale=.1)
    lens = np.cumsum(clusterSizes)
    dataSet[:lens[0]] +=clusterCenters[0]
    for i in range(numClusters-1):
        dataSet[lens[i]:lens[i+1]] += clusterCenters[i+1]
    return dataSet
    
    
    
def runSimBrute(args):
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    kMax,maxPulls,seed = args
    dataSet = createDataset(seed)
    
#     start = time.time()
#     maxBrutePull = 10000
    k,itersUsed,WkArr,WkbArr = gapStatBrute(dataSet,kMax,maxPulls,seed)
#     totalTime = (time.time()-start)*scipy.special.comb(n,d+1)/maxBrutePull

    with open(r"pkls_gap/naive_i{}_p{}_exp{}.pkl".format(setup,maxPulls,seed), "wb" ) as writeFile:
#         pickle.dump({"k":k,"itersUsed":itersUsed,"WkArr":WkArr,"WkbArr":WkbArr
#                     ,"dataSet":dataSet}, writeFile )
        pickle.dump({"k":k,"itersUsed":itersUsed}, writeFile )
        
def runSimAda(args):
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    kMax,maxPulls,const,seed = args
    dataSet = createDataset(seed)
    
    k,itersUsed,WkArr,WkbArr,numPulls = gapStat(dataSet,kMax,maxPulls,const,seed)

    with open(r"pkls_gap/ada_i{}_c{}_exp{}.pkl".format(setup,const,seed), "wb" ) as writeFile:
#         pickle.dumps({"k":k,"itersUsed":itersUsed
#                     ,"WkArr":WkArr,"WkbArr":WkbArr,"numPulls":numPulls
#                     ,"dataSet":dataSet}, writeFile )
        pickle.dump({"k":k,"itersUsed":itersUsed}, writeFile )
    


setup=6
maxPulls=3
kMax = 10

numTrials = 50
# constArr = [1,5,10]
constArr=[1]

pool      = mp.Pool(processes=50)

# arg_tuple = itertools.product([kMax],[maxPulls], constArr, list(range(numTrials)))
# for _ in tqdm(pool.imap_unordered(runSimAda, arg_tuple), total=numTrials*len(constArr)):
# 	pass


arg_tuple = itertools.product([kMax],[maxPulls], list(range(numTrials)))
for _ in tqdm(pool.imap_unordered(runSimBrute, arg_tuple), total=numTrials):
	pass