import itertools
import multiprocessing as mp
import numpy as np
import os, pickle
from tqdm import tqdm
import numpy.linalg as npl
import scipy.special,time
from scipy.special import comb as choose

from adaDataDepth import ruts2dDepth, ada_simplicial_median

def ceil(arr):
    return np.ceil(arr).astype(int)
        

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
    
    (i,n,d,epsilon,delta) = args

    ## set random seed to i//2 to check best arm
    np.random.seed(i//2)
    ## set random seed to 0 to run on same dataset, to obtain Figure 1b
    # np.random.seed(0)

    dataSet = np.random.normal(size=(n,d))
    np.random.seed(i) ## random seed reset to i to allow for deterministic running of bandit algorithm


    start = time.time()
    idx,estMeans,numPulls,numExactComp = ada_simplicial_median(dataSet,epsilon,delta)
    totalTime = time.time()-start

    with open(r"pkls/adaSim_{}_{}.pkl".format(n,i), "wb" ) as writeFile:
    
        pickle.dump([i//2,i,n,idx,numExactComp,numPulls,estMeans,totalTime,
                 "dataSeed","trial num","n","best arm","numExactComp","numPulls","estMeans","time"], writeFile )
        
        
def runSimBrute(args):
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    (n,d,_,_) = args
    
    np.random.seed(0) #### dataset doesn't matter, fixed amount of computation
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
    
    (n,d,_,_) = args
    
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
if __name__ == '__main__':
    mp.freeze_support()

    try:
        os.mkdir('pkls') ## save all files to pkls directory
    except:
        pass
    ### inputs for processing
    num_jobs = 3 ### for parallelization

    numTrials = 2 ### number of trials to perform for each value of n, default 50

    numN = 20 ### total number of values of n to try, default 20
    minN = 200 ### minimum n, default 200
    maxN = 20000 ### maximum n, default 20000

    delta = .001 ### allowable error probability
    epsilon=0 ### allowable depth error: for exact computation set as 0
    d=2 ### dimension of data

    nArr = np.round(np.logspace(np.log2(minN),np.log2(maxN),num=numN,base=2)).astype(int)

    ### parallelize over num_jobs threads
    pool      = mp.Pool(processes=num_jobs)
    arg_tuple = itertools.product(list(range(numTrials)),nArr,[d],[epsilon],[delta])
    for _ in tqdm(pool.imap_unordered(runSimAda, arg_tuple), total=numTrials*len(nArr)):
    	pass

    arg_tuple = itertools.product(nArr,[d],[epsilon],[delta])
    for _ in tqdm(pool.imap_unordered(runSimBrute, arg_tuple), total=len(nArr)):
    	pass

    arg_tuple = itertools.product(nArr,[d],[epsilon],[delta])
    for _ in tqdm(pool.imap_unordered(runSimRuts, arg_tuple), total=len(nArr)):
    	pass