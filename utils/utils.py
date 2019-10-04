import itertools
import numpy as np

def labelByThreshold(probabilities, threshold,eventLabel,noEventLabel):
    labels = []
    for prob in probabilities:
        if prob > threshold:
            labels.append(eventLabel)
        else:
            labels.append(noEventLabel)

    return np.asarray(labels)    


def diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

def splitCombinations(arr,minRange,maxRange): #Return the differente split combinations of a list between two range limits
    splits = []
    for j in range(minRange,maxRange+1):
        for i in list(itertools.combinations(arr,j)):
            set0 = diff(arr,i)
            set1 = list(i)
            splits.append((set0,set1))
    return splits