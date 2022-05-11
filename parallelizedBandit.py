import itertools
import multiprocessing as mp
import numpy as np
import os, pickle
from tqdm import tqdm, trange
import math
import numpy.linalg as npl
import scipy.special,time, scipy.stats
from collections import Counter
from scipy.special import comb as choose

def ceil(arr):
    return np.ceil(arr).astype(int)

def floor(arr):
    return np.floor(arr).astype(int)

#### Rousseuw and Ruts
# helpful ref https://github.com/olgazasenko/ColourfulSimplicialDepthInThePlane/blob/master/src/colourfulsimplicialdepthintheplane/RousseeuwAndRuts.java
def ruts2dDepth(data,point):
    (n,d) = data.shape
    pi =np.pi
    
    alphaArr = np.zeros(n)
    for i in range(n):
        u = (data[i]-point)/np.linalg.norm(data[i]-point)
        alphaArr[i] = np.arctan2(u[1],u[0])
    
    alphaArr = np.sort(alphaArr)
    maxGap = max(2*np.pi+alphaArr[0] - alphaArr[-1],np.max(alphaArr[1:]-alphaArr[:-1]))
    if maxGap>np.pi:
        return choose(n,2)/choose(n+1,3)
    alphaArr-= alphaArr[0]
    
    nu = max(np.where(alphaArr<np.pi)[0])+1

    betaArr = np.copy(alphaArr)-pi
    betaArr[alphaArr<pi] += 2*pi

    betaArr = np.sort(betaArr)
    
    mergedArr,wArr = mergeArrays(alphaArr,betaArr)
    
    hi = np.zeros(n)
    hi[0]=nu-1
    startLoc = min(np.where(mergedArr>pi)[0])

    
    NF = nu
    t = 1
    i = startLoc
    while i!= startLoc-1:
        if wArr[i]==1:
            NF+=1
        else:
            hi[t] = NF-(t+1)
            t+=1
        i+=1
        i %= 2*n
        
    hiCheck = np.zeros(n)
    j=0
    for i in range(n):
        while True:
            if (alphaArr[i] <= alphaArr[j] and alphaArr[j]< alphaArr[i] + pi) or (alphaArr[j] < alphaArr[i]-pi):
                j+=1
                j%=n
            else:
                j-=1
                break
        
        if j < i:
            hiCheck[i] = j-i+n
        else:
            hiCheck[i]=j-i

    output = choose(n,3)
    for i in range(n):
        output-= choose(hiCheck[i],2)
    
    return (output+choose(n,2))/choose(n+1,3)
    

    
def mergeArrays(arr1, arr2):
    n = len(arr1)
    arr3 = np.zeros(2*n)
    i = 0
    j = 0
    k = 0
    w = np.zeros(2*n)
    
    # Traverse both array
    while i < n and j < n:
        if arr1[i] < arr2[j]:
            arr3[k] = arr1[i]
            w[k]=1
            k = k + 1
            i = i + 1
            
        else:
            arr3[k] = arr2[j]
            w[k]= -1
            k = k + 1
            j = j + 1
            
 
    # Store remaining elements
    # of first array
    while i < n:
        arr3[k] = arr1[i];
        w[k]=1
        k = k + 1
        i = i + 1
        
 
    # Store remaining elements
    # of second array
    while j < n:
        arr3[k] = arr2[j];
        w[k]=-1
        k = k + 1
        j = j + 1
        
    return arr3,w


### adaptive simplicial median computation
def bestArm(dataSet,eps,delta):
    n,d = dataSet.shape
    r=1
    activeArmsArr = np.ones(n,dtype=bool)
    activeArmsSet = set(range(n))
    estMeans = np.zeros(n)
    numPulls = np.zeros(n)
    trPrev = 0
    
    
    while True:
        epsR = (1/2)**r
        trTotal = ceil(.2*epsR**(-2)*np.log(4*n*r**2/delta))
        tr=trTotal - trPrev
        trPrev = trTotal
        
        nr = len(activeArmsSet)
        tr = int(min(tr,scipy.special.comb(n,d+1)))
        
        threshold = int(n*np.log2(n))
        if tr>threshold and d==2:
            exactComputeIdx = list(activeArmsSet)
            exactComputeVal = np.zeros(len(exactComputeIdx))
            for (i,idx) in enumerate(exactComputeIdx):
                exactComputeVal[i] = ruts2dDepth(np.delete(dataSet,idx,axis=0),dataSet[idx])
                estMeans[idx]=exactComputeVal[i]
                numPulls[idx] += threshold
            return exactComputeIdx[exactComputeVal.argmax()],estMeans,numPulls,len(activeArmsSet)
        
        
        #### pull arms
        Yr = np.zeros((len(activeArmsSet),tr))
        myit = itertools.combinations(range(n),d+1)
        for j in range(tr):
            refIdxs = np.random.choice(n,size=d+1,replace=False)
            
            if tr==scipy.special.comb(n,d+1):
                refIdxs = next(myit)
            
            refPts = dataSet[refIdxs]
            X = refPts[:d].T - np.outer(refPts[-1], np.ones(d))
            
            ### batch solve linear systems
            queryPts = dataSet[activeArmsArr].T
            lamCoords = np.zeros((d+1,nr))
            lamCoords[:d] = npl.solve(X,queryPts-np.outer(refPts[-1],np.ones(nr)))

            lamCoords[-1] = 1-lamCoords.sum(axis=0)
            inSimplex = 1.0*np.all(lamCoords>-1*10**(-12),axis=0)

    
            Yr[:,j] = inSimplex
            
        estMeans[activeArmsArr] = (numPulls[activeArmsArr]*estMeans[activeArmsArr] + Yr.sum(axis=1))/(numPulls[activeArmsArr]+tr)
        numPulls[activeArmsArr]+=tr
        ### end pull arms
        
        maxVal = estMeans[activeArmsArr].max()
        newActive = set()
        activeArmsArr=np.zeros(n,dtype=bool)

        ### determine active arms for next round
        for i in activeArmsSet:
            if estMeans[i]> maxVal-epsR:
                newActive.add(i)
                activeArmsArr[i]=True
        activeArmsSet = newActive

        if len(activeArmsSet)==1 or epsR <= eps/2 or tr>=scipy.special.comb(n,d+1):
            return np.argmax(estMeans+100*activeArmsArr),estMeans,numPulls,0
    
        r+=1
        
        
        
### brute force simplicial median computation
def bruteForce(dataSet,maxPulls = -1):
    (n,d) = dataSet.shape
    estMeans = np.zeros(n)
    
    evalPts = np.logspace(0,np.log2(scipy.special.comb(n,d+1)),num=10,base=2,endpoint=True,dtype=int)
    if maxPulls != -1:
        evalPts = np.logspace(0,np.log2(maxPulls),num=10,base=2,endpoint=True,dtype=int)
    evalArgmax = np.zeros(len(evalPts))
    evalCtr = 0
    
    
    endIter = scipy.special.comb(n,d+1)
    if maxPulls!= -1:
        endIter = maxPulls
    
    for i,tup in enumerate(itertools.combinations(range(n),d+1)):
        refPts = dataSet[list(tup)]
        if maxPulls!= -1:
            refPts = dataSet[np.random.choice(n,d+1,replace=False)]
        
        X = refPts[:d].T - np.outer(refPts[-1], np.ones(d))

        queryPts = dataSet.T
        lamCoords = np.zeros((d+1,n))
        lamCoords[:d] = npl.solve(X,queryPts-np.outer(refPts[-1],np.ones(n)))

        lamCoords[-1] = 1-lamCoords.sum(axis=0)
        inSimplex = 1.0*np.all(lamCoords>-1*10**(-12),axis=0)

        estMeans += inSimplex
        
        if i==maxPulls:
            break
        if i==evalPts[evalCtr]:
            evalArgmax[evalCtr] = np.argmax(estMeans)
            evalCtr+=1
        
    evalArgmax[-1] = np.argmax(estMeans)
        
    if maxPulls!= -1:
        return evalArgmax,evalPts*n,estMeans/maxPulls
    return evalArgmax,evalPts*n,estMeans/scipy.special.comb(n,d+1)



def runSimAda(args):
    ### to ensure that python doesn't try to multithread each run
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    (i,n) = args
    
    np.random.seed(i//2)

    dataSet = np.random.normal(size=(n,d))
    np.random.seed(i)


    start = time.time()
    idx,estMeans,numPulls,numExactComp = bestArm(dataSet,epsilon,delta)
    totalTime = time.time()-start

    with open(r"pkls/adaSim_{}_{}.pkl".format(n,i), "wb" ) as writeFile:
    
        pickle.dump([i//2,i,n,idx,numExactComp,numPulls,estMeans,totalTime,
                 "dataSeed","trial num","n","best arm","numExactComp","numPulls","estMeans","time"], writeFile )
        
        
def runSimBrute(args):
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    n = args
    
    np.random.seed(0)
    dataSet = np.random.normal(size=(n,d))


    start = time.time()
    maxBrutePull = 10000
    evalArgmax,evalPts,estMeansBrute = bruteForce(dataSet,maxPulls=maxBrutePull)
    totalTime = (time.time()-start)*scipy.special.comb(n,d+1)/maxBrutePull

    with open(r"pkls/bruteSim_{}.pkl".format(n), "wb" ) as writeFile:
        pickle.dump({"time":totalTime}, writeFile )
        
def runSimRuts(args):
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    n = args
    
    np.random.seed(0)
    dataSet = np.random.normal(size=(n,d))


    start=time.time()
    bruteCompNum = 50
    rutsDepth = np.zeros(n)
    for j in range(bruteCompNum):
        rutsDepth[j] = ruts2dDepth(np.delete(dataSet,j,axis=0),dataSet[j])
    totalTime= (time.time()-start)*n/bruteCompNum

    with open(r"pkls/rutsSim_{}.pkl".format(n), "wb" ) as writeFile:
        pickle.dump({"time":totalTime}, writeFile )
    


### actual body of code to be run

### inputs for processing
delta = .001
epsilon=.000001 ## irrelevant for exact computation, just set as small value
nArr = np.round(np.logspace(np.log2(200),np.log2(20000),num=20,base=2)).astype(int)
numTrials = 20
num_jobs = 50
d=2

### parallelize over num_jobs threads
pool      = mp.Pool(processes=num_jobs)
arg_tuple = itertools.product(list(range(numTrials)),nArr)
for _ in tqdm(pool.imap_unordered(runSimAda, arg_tuple), total=numTrials*len(nArr)):
	pass

for _ in tqdm(pool.imap_unordered(runSimBrute, nArr), total=len(nArr)):
	pass

for _ in tqdm(pool.imap_unordered(runSimRuts, nArr), total=len(nArr)):
	pass


