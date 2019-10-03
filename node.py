
class Node():

    def __init__(self,decisionRule,criterionValue,countValues):
        self.leftChild = None
        self.rightChild = None
        self.decisionRule = decisionRule
        self.criterionValue = criterionValue
        self.countValues = countValues
    
    def appendLeftChild(self,child):
        self.leftChild=child

    def appendRightChild(self,child):
        self.leftChild=child
    