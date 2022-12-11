import itertools
import numpy as np
import numpy.linalg as npl
import scipy.special
from scipy.special import comb as choose

def ceil(arr):
    return np.ceil(arr).astype(int)

#### Rousseuw and Ruts
# helpful ref https://github.com/olgazasenko/ColourfulSimplicialDepthInThePlane/blob/master/src/colourfulsimplicialdepthintheplane/RousseeuwAndRuts.java
def ruts2dDepth(data,point):
    (n,d) = data.shape
    pi =np.pi

    #### helper function
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
    
    ### 2d data depth computation
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
    


### Adaptive simplicial median computation
### See Algorithm 1 of "Adaptive Data Depth via Multi-Armed Bandits" by Baharav and Lai 2022 for details
### Inputs:
#### dataSet: n x d matrix of n data points in d dimensions
#### delta: failure probability bound, by default 1%
#### eps: accuracy to which simplicial median should be computed
def ada_simplicial_median(dataSet,eps=0,delta=.01):
    n,d = dataSet.shape
    r=1 ## round counter
    activeArmsArr = np.ones(n,dtype=bool) ## candidate simplicial median
    activeArmsSet = set(range(n)) 
    estMeans = np.zeros(n) ## estimated simplicial depth of points
    numPulls = np.zeros(n) ## number of random simplices computed per point
    trPrev = 0
    
    
    while True:
        epsR = (1/2)**r ## accuracy to which we should estimate in round r
        trTotal = ceil(.2*epsR**(-2)*np.log(4*n*r**2/delta))
        tr=trTotal - trPrev
        trPrev = trTotal
        
        nr = len(activeArmsSet)
        tr = int(min(tr,scipy.special.comb(n,d+1))) ## number of pulls to make
        
        ## currently specialized for d=2, can be extended to other exact computation methods
        threshold = int(n*np.log2(n))
        if tr>threshold and d==2: ## call specialized exact computation method, 2d
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
            
            if tr==scipy.special.comb(n,d+1): ## exact computation, iterate over all combinations
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
        
        
        