import itertools
import numpy as np

from ../DecisionRule import DecisionRule
from ../Node import Node

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

''' Method that create a Node with minimum split entropy for a list of combinations in a dataframe of data '''
def nodeForMinEntropy(splits,df):
    n_lambda = df.shape[0]
    #entropies={}
    minEntropy = 1 #Max theorical entropy for a binary target value
    selectedSplit = None
    
    selectedVariable = None
    for variable in splits:
        for split in splits[variable]:
            type1 = split[0]
            type2 = split[1]
            subset1 = df[df[variable].isin(type1)]
            n_lambda1 = subset1.shape[0]
            n_lambda_event1 = subset1[subset1['target']==1].shape[0]
            n_lambda_nonEvent1 = subset1[subset1['target']==0].shape[0]

            subset2 = df[df[variable].isin(type2)]
            n_lambda2 = subset2.shape[0]
            n_lambda_event2 = subset2[subset2['target']==1].shape[0]
            n_lambda_nonEvent2 = subset2[subset2['target']==0].shape[0]
 
            if((n_lambda_event1/n_lambda1)>0):
                entropi1 = -(n_lambda_event1/n_lambda1)*np.log2(n_lambda_event1/n_lambda1)-(n_lambda_nonEvent1/n_lambda1)*np.log2(n_lambda_nonEvent1/n_lambda1)
            else:
                entropi1 = 0
            if((n_lambda_event2/n_lambda2)>0):
                entropi2 = -(n_lambda_event2/n_lambda2)*np.log2(n_lambda_event2/n_lambda2)-(n_lambda_nonEvent2/n_lambda2)*np.log2(n_lambda_nonEvent2/n_lambda2)
            else:
                entropi2 = 0

            split_entropy=entropi1*(n_lambda1/n_lambda)+entropi2*(n_lambda2/n_lambda)

            if(split_entropy<minEntropy):
                minEntropy = split_entropy
                selectedSplit = split
                selectedVariable = variable
            #entropies["{}|{}".format(type1,type2)]=split_entropy
    #min_split = min(entropies, key=entropies.get)    
    #print("Min Split Entropy: {}; Split Entropy: {}".format(min_split,entropies[min_split]))

    decisionRule = DecisionRule(selectedVariable,selectedSplit)
    criterionValue = minEntropy
    countValues = (df[df['target']==1].shape[0],df[df['target']==0].shape[0])
    return Node(decisionRule,criterionValue,countValues)


def createNodeFromCriterion(splits,df,criterion):

    if(criterion.lower() == 'entropy'):
        return nodeForMinEntropy(splits,df)
         