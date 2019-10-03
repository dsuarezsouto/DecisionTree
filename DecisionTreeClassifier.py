from utils.LabelByThreshold import labelByThreshold
from _tree import Tree
class DecisionTreeClassifier():

    def __init__(self,criterion='entropy',max_depth=3,threshold=0.5,ordinalFeatures=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.threshold = threshold
        self.ordinalFeatures = ordinalFeatures
    

    def getParams(self):
        return self.__dict__

    def train(self,X,y):
        self.tree = Tree()
        return self.tree.build(X,y,self.ordinalFeatures)

    def predict(self,X):

        if(self.tree==None):
            raise Exception("This instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        #Get probabilities
        probabilities = self.tree.predict(X)
        #Label the data comparing with threshold
        eventLabel = self.tree.getEventLabel()
        noEventLabel = self.tree.getNoEventLabel()
        labels = labelByThreshold(probabilities,self.threshold,eventLabel,noEventLabel)
        return labels

    def predict_proba(self,X):
        if(self.tree==None):
            raise Exception("This instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

        probabilities = self.tree.predict(X)

        return probabilities