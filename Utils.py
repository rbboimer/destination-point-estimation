# -*- coding: utf-8 -*-
import numpy as np
  
def getEstimatedEndLocations(drivingData, testData, rho, weights):
    """
    Get the most probable destination point for each row of the test dataset
    
    Args:
        drivingData: training dataset
        testData: test dataset
        rho: evaporation coefficient
        weights: array of scaled weighting factors (Z)
    
    Returns:
        An array with the most probable destinations
    """
    
    # endLocNames is a list containing the possible end locations
    # for the toy example this would be ['other', 'work', 'gym', 'home']
    endLocNames = list(set(testData["EndLoc"]))
    estimatedEndLocs = []
        
    for testDataRow in range(0, len(testData)):
        loc = getMostProbableDestination(endLocNames, drivingData, testData[testDataRow:testDataRow+1], rho, weights)
        estimatedEndLocs.append(loc) 
    return estimatedEndLocs


def getMostProbableDestination(endLocNames, drivingData, testDataRow, rho, weights):
    """
    Get the most probable destination point
    
    Args:
        endLocNames: list containing the possible end locations
        drivingData: training dataset
        testDataRow: one row of the test dataset
        rho: evaporation coefficient
        weights: array of scaled weighting factors (Z)
    
    Returns:
        The name of the most probable destination location
    """
    
    # the number of unique end locations (4 for the toy example)
    locCount = len(endLocNames)
    currentLoc = testDataRow["StartLoc"]
    
    locFactors = np.zeros(locCount)
    for i in range(0, locCount):
        # avoid testing the probability of heading to location x when currently in location x
        if np.sum(currentLoc == endLocNames[i]):
            locFactors[i] = -1
        else:
            locFactors[i] = calcTransitionMatrixFactor(drivingData, testDataRow, endLocNames[i], rho, weights)

    return endLocNames[np.argmax(locFactors)]
   

def calcTransitionMatrixFactor(drivingData, testDataRow, destination, rho, weights):
    """
    Generates the matrix factor for the given destination point
    
    Args:
        drivingData: training dataset
        testDataRow: one row of the test dataset
        destination: destination location (ex: 'work')
        rho: evaporation coefficient
        weights: array of scaled weighting factors (Z)
    
    Returns:
        The transition matrix factor
    """
    
    countDay, countStartLoc, countNumPassengers, countHoliday, countTime = 0, 0, 0, 0, 0
    countDayTotal, countStartLocTotal, countHolidayTotal, countNumPassengersTotal, countTimeTotal = 0, 0, 0, 0, 0

    Day = testDataRow["Day"].tolist()[0]
    StartLoc = testDataRow["StartLoc"].tolist()[0]
    Holiday = testDataRow["Holiday"].tolist()[0]
    Time = testDataRow["Time"].tolist()[0]
    NumPassengers = testDataRow["NbPassengers"].tolist()[0]
    
    for i in range(0, len(drivingData)):
        # endLocMatch = true if the current location is the same as the destination
        endLocMatch = drivingData["EndLoc"][i] == destination
        
        if endLocMatch and drivingData["Day"][i] == Day:
            countDay = countDay + weights[i]
            countDayTotal = countDayTotal + weights[i]
        elif drivingData["Day"][i] == Day:
            countDayTotal = countDayTotal + weights[i]
            
        if endLocMatch and drivingData["StartLoc"][i] == StartLoc:
            countStartLoc = countStartLoc + weights[i]
            countStartLocTotal = countStartLocTotal + weights[i]
        elif drivingData["StartLoc"][i] == StartLoc:
            countStartLocTotal = countStartLocTotal + weights[i]
        
        if endLocMatch and drivingData["Holiday"][i] == Holiday:
            countHoliday = countHoliday + weights[i]    
            countHolidayTotal = countHolidayTotal + weights[i] 
        elif drivingData["Holiday"][i] == Holiday:
            countHolidayTotal = countHolidayTotal + weights[i]         

        if endLocMatch and drivingData["NbPassengers"][i] == NumPassengers:
            countNumPassengers = countNumPassengers + weights[i]
            countNumPassengersTotal = countNumPassengersTotal + weights[i]  
        elif drivingData["NbPassengers"][i] == NumPassengers:
            countNumPassengersTotal = countNumPassengersTotal + weights[i]  
        
        if endLocMatch and drivingData["Time"][i] == Time:
            countTime = countTime + weights[i]
            countTimeTotal = countTimeTotal + weights[i]
        elif drivingData["Time"][i] == Time:
            countTimeTotal = countTimeTotal + weights[i]           
    
    if countDayTotal > 0 and countStartLocTotal > 0 and countHolidayTotal > 0 and countNumPassengersTotal > 0 and countTimeTotal > 0:
        # equation 4
        return (countDay/countDayTotal) * (countStartLoc/countStartLocTotal) * (countHoliday/countHolidayTotal) * (countNumPassengers/countNumPassengersTotal) * (countTime/countTimeTotal)
    return 0


def convertTimeToClasses(timeCol, timeSegmentSplit):
    """
    Convert time from a qualitiative feature to a quantitative feature
    
    Args:
        timeCol: the dataset time column in qualitative form
        timeSegmentSplit: the range (in minutes) that defines each bucket of time
    
    Returns:
        an updated time column represented as a quantitative feature (an array
        of integers representing buckets of length timeSegmentSplit)
    """
    timeColQuant = np.zeros(len(timeCol))
    
    for i in range(len(timeCol)):
        timeQuant = timeCol.iloc[i]
        # calculate the time as the time elapsed since midnight (in minutes)
        minutesFromMidnight = np.fix(timeQuant) * 60 + np.fix((timeQuant - \
                                    np.fix(timeQuant)) * 100)
        # convert the time to the class value
        timeColQuant[i] = np.round(minutesFromMidnight/timeSegmentSplit)
    
    return timeColQuant


def getAccuracyArray(estimatedEndLocs, testEndLocs):
    """
    Calculate the accuracy (%) between the destination location in the test
    dataset and the estimated desgination location
    
    Args:
        estimatedEndLocs: array with the most probable destinations
        testEndLocs: array with the actual (test data) destinations
    
    Returns:
        an array of percentage representing the accuracy
    """
    
    numCorrectPredictions = np.sum(testEndLocs == estimatedEndLocs)
    return numCorrectPredictions / len(estimatedEndLocs)     


def getIndexOfFirstRecordingToKeep(dataSetLength, rho):
    """
    Calculate the evaporation rate over the entire dataset and provide the 
    index of the first recording for which evaporation rate is not 0
    
    Args:
        dataSetLength: the number of recordings in the training dataset
        rho: evaporation coefficient
    
    Returns:
        an index of the first recording to keep
    """
    k = np.arange(dataSetLength, 0, -1)
    er = -np.exp(k * rho) + 2 # equation 7
    
    # Set the bound of evaporation rate (er) to [0,1]
    superThresholdIndices = er < 0
    er[superThresholdIndices] = 0
    superThresholdIndices = er > 1
    er[superThresholdIndices] = 1
    
    # Return the first non-zero index
    return np.nonzero(er)[0][0]


def getWeights(dataSetLength, rho):
    """
    Calculate the weight factors (Z) for the dataset
    
    Args:
        dataSetLength: the number of recordings in the training dataset
        rho: evaporation coefficient
    
    Returns:
        an array of scaled weight factors for each recording in the dataset
    """
    k = np.arange(dataSetLength, 0, -1)
    er = -np.exp(k * rho) + 2 # equation 7
    
    Z = np.round(dataSetLength * er) # equation 8
    Z = Z.astype(np.int64)
    
    # avoid unnecessary multiplication by standardizing to start at the lowest factor
    Z = Z - np.min(Z) + 1
    return Z