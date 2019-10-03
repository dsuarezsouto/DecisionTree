import pandas as pd


class Tree():


    def build(self,X,y,ordinalFeatures):
        #Todo: Build Tree searching the decision rules, etc
        self.eventLabel = None
        self.noEventLabel = None
        if(isinstance(X)==pd.DataFrame):
            # Check if there are Categorical features
        else:
                
        return self
    
    def getEventLabel(self):
        return self.eventLabel
    
    def getNoEventLabel(self):
        return self.getNoEventLabel
    
    def predict(self,X):

