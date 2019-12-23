import itertools
import numpy as np

from DecisionRule import DecisionRule
from node import Node


def labelByThreshold(probabilities, threshold):
    labels = []
    for prob in probabilities:
        if prob >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return np.asarray(labels)    


def diff(li1, li2): 
    return list(set(li1) - set(li2))


''' Return the differente split combinations of a list between two range limits'''
def splitCombinations(arr, minRange, maxRange):
    splits = []
    for j in range(minRange,maxRange+1):
        for i in list(itertools.combinations(arr, j)):
            set0 = diff(arr, i)
            set1 = list(i)
            splits.append((set0, set1))
    return splits


''' Method that create a Node with minimum split entropy for a list of combinations in a dataframe of data '''
def nodeForMinEntropy(splits, df):

    n_lambda = df.shape[0]

    min_entropy = 1  # Max theoretical entropy for a binary target value
    selected_split = None
    
    selected_variable = None
    for variable in splits:
        for split in splits[variable]:
            type1 = split[0]
            type2 = split[1]

            subset1 = df[df[variable].isin(type1)]
            n_lambda1 = subset1.shape[0]

            n_lambda_event1 = subset1[subset1['target'] == 1].shape[0]
            n_lambda_non_event1 = subset1[subset1['target'] == 0].shape[0]

            subset2 = df[df[variable].isin(type2)]
            n_lambda2 = subset2.shape[0]
            n_lambda_event2 = subset2[subset2['target'] == 1].shape[0]
            n_lambda_non_event2 = subset2[subset2['target'] == 0].shape[0]

            if n_lambda1 == 0 or n_lambda2 == 0:
                continue

            if (n_lambda_event1/n_lambda1) > 0:
                entropy1 = -(n_lambda_event1/n_lambda1) * np.log2(n_lambda_event1/n_lambda1) -\
                           (n_lambda_non_event1/n_lambda1) * np.log2(n_lambda_non_event1/n_lambda1)
            else:
                entropy1 = 0
            if (n_lambda_event2/n_lambda2)>0 :
                entropy2 = -(n_lambda_event2/n_lambda2) * np.log2(n_lambda_event2/n_lambda2) -\
                           (n_lambda_non_event2/n_lambda2) * np.log2(n_lambda_non_event2/n_lambda2)
            else:
                entropy2 = 0

            split_entropy = entropy1 * (n_lambda1/n_lambda) + entropy2 * (n_lambda2/n_lambda)

            if split_entropy<min_entropy:
                min_entropy = split_entropy
                selected_split = split
                selected_variable = variable

    decision_rule = DecisionRule(selected_variable,selected_split)
    criterion_value = min_entropy
    count_values = (df[df['target']==1].shape[0], df[df['target'] == 0].shape[0])
    return Node(decision_rule,criterion_value,count_values,splits)


def createNodeFromCriterion(splits, df, criterion):

    if criterion.lower() == 'entropy':
        return nodeForMinEntropy(splits,df)
