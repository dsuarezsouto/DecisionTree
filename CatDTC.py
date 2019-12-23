from utils import labelByThreshold
from tree import Tree


class CatDTC():

    def __init__(self,criterion='entropy',max_depth=3, threshold=0.5, ordinalFeatures=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.threshold = threshold
        self.ordinalFeatures = ordinalFeatures
        self.tree = None

    def get_params(self):
        return self.__dict__

    def fit(self, X, y):
        self.tree = Tree()
        self.tree.build(X,y,self.ordinalFeatures,self.max_depth,self.criterion)

    def predict(self, X):

        if self.tree is None:
            raise Exception("This instance is not fitted yet. Call 'fit' with appropriate arguments "
                            "before using this method.")

        # Get probabilities
        probabilities = self.tree.predict(X)

        labels = labelByThreshold(probabilities, self.threshold)
        return labels

    def predict_proba(self,X):
        if self.tree is None:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

        probabilities = self.tree.predict(X)

        return probabilities
