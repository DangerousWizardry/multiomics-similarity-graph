from scipy.stats import norm
from tqdm.auto import tqdm
import time
import pandas as pd
import numpy as np


# Configuration constant
SAMPLED_MIN = -20
SAMPLED_MAX = 20
SAMPLED_DELTA = 0.001

# Setting up normal distribution
sampled_normal_distribution = np.array([norm.cdf(i) for i in np.arange(SAMPLED_MIN,SAMPLED_MAX,SAMPLED_DELTA)])
sampled_abs_correction=abs(SAMPLED_MIN)
sampled_rel_correction=1/SAMPLED_DELTA

# Getting the index in the sampled_normal_distribution where alpha risk is overtake
def getSampledRangeIndex(alpha=0.05):
    idxAlphaMin = 0
    idxAlphaMax = int(sampled_abs_correction * sampled_rel_correction)
    while sampled_normal_distribution[idxAlphaMin] < alpha:
        idxAlphaMin+=1
    while sampled_normal_distribution[idxAlphaMax] < 1- alpha:
        idxAlphaMax+=1
    return (idxAlphaMin,idxAlphaMax)

# Return an array of fitted normal law from each column in the dataset
def getNormalLawsFromDataset(dataset):
    if len(dataset.shape) > 1:
        normalLaws = []
        for column in dataset:
            mu, std = norm.fit(dataset[column])
            normalLaws.append({"type":"normal","mu":mu,"std":std})
        return normalLaws
    print("Incorrect shape of dataset")

# test similarity of a list of value (x1) against x2 using a given law & alpha risk (represented by idxAlphaMin,idxAlphaMax)
# only normal law is implemented for the moment (we use sampled normal law to increase performance)
def testSimilarity(x1,x2,law,idxAlphaMin,idxAlphaMax):
    if law["type"] == "normal": 
        # start_time = time.time()
        # plist = scipy.special.ndtr((x1-x2)/law["std"])
        # print("scipy --- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        keys = (((x1-x2)/law["std"]) + sampled_abs_correction ) * sampled_rel_correction
        # print("custom --- %s seconds ---" % (time.time() - start_time))
        # print(min(keys))
        # print(max(keys))
        plist = [abs(sampled_normal_distribution[int(key)]-1) if sampled_normal_distribution[int(key)]>0.5 else sampled_normal_distribution[int(key)] for key in keys]
        return np.array(plist)
    else:
        print("Not implemented for law type: ")

# Generate a 3D similarity matrix (column,row,row) with a given dataset assuming each column is normally distributed
def getSimilarityMatrix(dataset,alpha=0.05):
    shape0 = dataset.shape[0]
    shape1 = dataset.shape[1]
    similarityMatrix = np.zeros((shape1,shape0,shape0,))
    laws = getNormalLawsFromDataset(dataset)
    idxAlphaMin,idxAlphaMax = getSampledRangeIndex(alpha)
    for columnidx in tqdm(range(len(dataset.columns))):
        for subjectaidx in range(shape0):
            x1 = dataset.values[:,columnidx]
            x2 = [dataset.values[subjectaidx,columnidx]] * shape0
            similarityMatrix[columnidx][subjectaidx][:] = testSimilarity(x1,x2,laws[columnidx],idxAlphaMin,idxAlphaMax)
    return similarityMatrix

