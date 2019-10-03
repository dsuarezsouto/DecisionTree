
import numpy as np

def labelByThreshold(probabilities, threshold,eventLabel,noEventLabel):
    labels = []
    for prob in probabilities:
        if prob > threshold:
            labels.append(eventLabel)
        else:
            labels.append(noEventLabel)

    return np.asarray(labels)    