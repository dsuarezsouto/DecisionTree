
class Node():

    def __init__(self,decisionRule,criterionValue,countValues,splits):
        self.leftChild = None
        self.rightChild = None
        self.decisionRule = decisionRule
        self.criterionValue = criterionValue
        self.countValues = countValues
        self.splits = splits
    
    def appendLeftChild(self,child):
        self.leftChild=child

    def appendRightChild(self,child):
        self.leftChild=child

    def getDecisionRule(self):
        return self.decisionRule
    def getSplits(self):
        return self.splits
    