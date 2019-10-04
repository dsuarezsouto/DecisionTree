import pandas as pd
from utils.utils import splitCombinations

class Tree():


    def build(self,X,y,ordinalFeatures):
        #Todo: Build Tree searching the decision rules, etc
        self.eventLabel = None
        self.noEventLabel = None
        if(isinstance(X)==pd.DataFrame):
            # Check if there are Categorical features
            categoricalPredictors = X.select_dtypes(include=['object','category']).columns
            if len(categoricalPredictors)>0:
                #Combinations of the Categorical values
                minRange = int(len(categoricalPredictors)/2)
                maxRange = len(categoricalPredictors)-1
                splits = dict((column, splitCombinations(categoricalPredictors,minRange,maxRange)) for column in categoricalPredictors)
            else:
                raise Exception("There are not categorical features in X dataset")
        else:
            raise Exception("X is not a dataframe")
                
        return self
    
    def getEventLabel(self):
        return self.eventLabel
    
    def getNoEventLabel(self):
        return self.getNoEventLabel
    
    def predict(self,X):

