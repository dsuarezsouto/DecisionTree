import pandas as pd
import numpy as np
from utils import splitCombinations, createNodeFromCriterion


class Tree():
    def __init__(self):
        self.rootNode = None
        self.splits = None

    def build(self, X, y, ordinalFeatures, max_depth, criterion):

        if isinstance(X,pd.DataFrame):

            # Check if there are Categorical features
            categoricalPredictors = X.select_dtypes(include=['object','category']).columns
            if len(categoricalPredictors) > 0:
                # Combinations of the Categorical values
                splits = dict()
                for column in categoricalPredictors:
                    splits[column] = splitCombinations(X[column].unique(),int(len(X[column].unique())/2),len((X[column].unique()))-1)

                self.splits = splits
                
                if issubclass(y.dtype.type,np.integer):
                    X['target'] = y
                    self.rootNode = createNodeFromCriterion(self.splits,X,criterion)

                    children = self.createChildren(X,self.rootNode,criterion)

                    for i in range(max_depth-2):
                        tmpChildren = []
                        for child in children:
                            tmpChildren.append(self.createChildren(child[1],child[0],criterion))
                        children = tmpChildren
                else:
                    raise Exception("Target values must be encoded (ex. '1' and '0')")  
            else:
                raise Exception("There are not categorical features in X dataset")
        else:
            raise Exception("X is not a dataframe")
                
        return self
    
    ''' Return a list of tuples with each child with its filtered X associated '''
    def createChildren(self, X, parentNode, criterion):
        # Filter df for each side
        leftValues = parentNode.getDecisionRule().getSplit()[0]
        rightValues = parentNode.getDecisionRule().getSplit()[1]
        columnFilter = parentNode.getDecisionRule().getVariable()
        splitsL = parentNode.getSplits().copy()
        splitsR = parentNode.getSplits().copy()

        # Left Side
        X_leftBranch = X[X[columnFilter].isin(leftValues)]
        splitsL[columnFilter] = splitCombinations(leftValues,int(len(leftValues)/2),len(leftValues)-1)

        children = []
        
        node = createNodeFromCriterion(splitsL,X_leftBranch,criterion)
        parentNode.leftChild = node
        children.append((node,X_leftBranch,splitsL))

        # Right Side
        X_rightBranch = X[X[columnFilter].isin(rightValues)]
        splitsR[columnFilter] = splitCombinations(rightValues,int(len(rightValues)/2),len(rightValues)-1)
        node = createNodeFromCriterion(splitsR,X_rightBranch,criterion)
        parentNode.rightChild = node
        children.append((node,X_rightBranch,splitsR))

        return children

    def predict(self, X):

        X['probabilities'] = X.apply(lambda row: self.get_label(row),axis=1)
        return X['probabilities'].values

    def get_label(self, row):
        node = self.rootNode
        while node.leftChild is not None or node.rightChild is not None:
            variable = node.getDecisionRule().getVariable()
            split_left = node.getDecisionRule().getSplit()[0]
            split_right = node.getDecisionRule().getSplit()[1]
            val_sample = row[variable]

            if val_sample in split_left:
                node = node.leftChild
            elif val_sample in split_right:
                node = node.rightChild
            else:
                raise Exception("Error: Found value in predictor "+variable+" that does not match in any of the "+
                                    "splits of the node: splitL {} | splitR:{}".format(split_left,split_right))

        n_event = node.countValues[0]
        n_no_event = node.countValues[1]

        return n_event / (n_event+n_no_event)



