'''
    Class for the DecisionRule of each node in the DT.
    @Params: 
        - variable: Column name of the split
        - split: Tuple with each values of the split
'''
class DecisionRule():

    def __init__(self,variable,split):
        self.variable = variable
        self.split = split
    
    def getVariable(self):
        return self.variable

    def getSplit(self):
        return self.split