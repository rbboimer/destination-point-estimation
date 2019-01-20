# Destination Point Estimation: A Markov Model Based Approach

Estimating the destination point or the intended travel destination is imperative for an improved passenger-vehicle experience. For example, this feature helps in improving power train efficiencies and electric range estimations. It can also help with focused content through targeted ads, or assist with road hazard alerts that enable preemptive situation mitigation. In this work, we present a Markov Model based approach that learns from driving history. The model utilizes only few features such as time and current location to predict the destination point. To mitigate the passengers' behavioral changes, we add an evaporation rate factor so that older recordings have less influence on the model than newer ones. The evaporation rate also allows for a more evolved model that avoids getting stuck in local minimums. Results show that with less than 100 recordings we are able to successfully predict more than 85\% of the travel destination points even with the incorporation of passengers' habitual changes.

## Getting Started

This repo includes sample training and test datasets as well as the code required to reproduce the results obtained in the accompanying paper.

### Prerequisites

* [python 3.6.x](https://www.python.org/downloads/release/python-368/)
* [pandas](https://pandas.pydata.org)
* [numpy](http://www.numpy.org)

### Running the code

To run the code save `DestinationPointEstimation.py`, `Utils.py`, `TestData.csv`, and `TrainData.csv` in the same directory and run `DestinationPointEstimation.py` in your preferred python environment. Once it is running you will see iteration and accuracy results printed to the console. 

```
Iteration  1 / 216 Accuracy is:  0.28
Iteration  2 / 216 Accuracy is:  0.3
...
```

Upon completion you will get a plot reflecting the final accuracy results per iteration (image cannot be added in anonymous mode).

# ![results plot](https://github.com/rbboimer/destination-point-estimation/blob/master/images/SampleResults.png)
