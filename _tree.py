import pandas as pd
from utils.utils import splitCombinations, createNodeFromCriterion

class Tree():


    def build(self,X,y,ordinalFeatures,max_depth,criterion):

        self.eventLabel = None
        self.noEventLabel = None
        if(isinstance(X)==pd.DataFrame):
            # Check if there are Categorical features
            categoricalPredictors = X.select_dtypes(include=['object','category']).columns
            if len(categoricalPredictors)>0:
                #Combinations of the Categorical values
                self.splits = dict((column, splitCombinations(X[column].unique(),int(len(X[column].unique())/2),len((X[column].unique())-1)) for column in categoricalPredictors)
                if(issubclass(y.dtype.type,np.integer)):
                    X['target']=y
                    rootNode = createNodeFromCriterion(splits,X,criterion)
                    children = createChildren(X,rootNode,criterion)
                    for i in range(max_depth-2):
                        tmpChildren = []
                        for child in children:
                            tmpChildren.append(createChildren(child[1],child[0],criterion))
                        children = tmpChildren
                else:
                    raise Exception("Target values must be encoded (ex. '1' and '0')")  
            else:
                raise Exception("There are not categorical features in X dataset")
        else:
            raise Exception("X is not a dataframe")
                
        return self
    
    def getEventLabel(self):
        return self.eventLabel
    
    def getNoEventLabel(self):
        return self.getNoEventLabel
    
    '''
        Return a list of tuples with each child with its filtered X associated
    '''
    def createChildren(self,X,parentNode,criterion):
        #Filter df for each side
        leftValues = parentNode.getDecisionRule().getSplit()[0]
        rightValues = parentNode.getDecisionRule().getSplit()[1]
        columnFilter = parentNode.getDecisionRule().getVariable()
        
        #Left Side
        X_leftBranch = X[X[columnFilter].isin(leftValues)]
        splits = self.splits
        splits[columnFilter] = leftValues
        children = []
        
        node = createNodeFromCriterion(splits,X_leftBranch,criterion)
        parentNode.leftChild = node
        children.append(tuple(node,X_leftBranch))

        #Right Side
        X_rightBranch = X[X[columnFilter].isin(rightValues)]
        splits[columnFilter] = rightValues
        node = createNodeFromCriterion(splits,X_rightBranch,criterion)
        parentNode.righChild = node
        children.append(tuple(node,X_rightBranch))

        return children
    def predict(self,X):


