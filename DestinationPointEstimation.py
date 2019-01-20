# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils import getIndexOfFirstRecordingToKeep
from Utils import getEstimatedEndLocations
from Utils import getAccuracyArray
from Utils import getWeights
from Utils import convertTimeToClasses


###############################################################################
'''
Step one: define tuneable parameters

rho: evaporation coefficient (how fast to evaporate old data)
     at rho = 0.0032 the recorded data influence diminishes after 150 recordings

timeSegmentSplit: the range (in minutes) that defines each bucket of time
                  timeSegmentSplit = 240 will result in 6 buckets
'''
rho = 0.0032
timeSegmentSplit = 240


###############################################################################
'''
Step two: read the driving (training) dataset and the test dataset
'''
drivingData = pd.read_csv('TrainData.csv', index_col=0)
testData    = pd.read_csv('TestData.csv', index_col=0)


###############################################################################
'''
Step three: delete data points older than x readings where x is calculated using
the evaporation coefficient
'''
firstValueToKeepIndex = getIndexOfFirstRecordingToKeep(len(drivingData), rho)

if firstValueToKeepIndex:
    drivingData = drivingData.drop(drivingData.index[0: firstValueToKeepIndex])

drivingData.reset_index(drop=True,inplace=True)


###############################################################################
'''
Step four: feature characterization

Approach one: split the quantitative feature (time) into segments so it can be
handled as if it were qualitiative.
'''
drivingData["Time"] = convertTimeToClasses(drivingData["Time"], timeSegmentSplit)
testData["Time"] = convertTimeToClasses(testData["Time"], timeSegmentSplit)


###############################################################################
'''
Step five: generate location estimate and accuracy array
'''
dataSetLength = len(drivingData)
accuracyArray = np.zeros(dataSetLength)
weights = getWeights(dataSetLength, rho) # calculate weight factors

for i in range(1, dataSetLength+1):
    estimatedEndLocs = getEstimatedEndLocations(drivingData[0:i], testData, rho, weights)
    accuracyArray[i-1] = getAccuracyArray(estimatedEndLocs, testData["EndLoc"])
    print("Iteration ", i, "/", dataSetLength, "Accuracy is: ", accuracyArray[i-1])


###############################################################################
'''
Step six: plot the accuracy results (iteration vs accuracy)
'''
plt.rcParams.update({'font.size': 16})
plt.plot(accuracyArray)
plt.xlabel("Drive Data Iteration")
plt.ylabel("Success Rate")