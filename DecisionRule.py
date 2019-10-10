class DecisionRule():

    def __init__(self,variable,split):
        self.variable = variable
        self.split = split
    
    def getVariable(self):
        return self.variable

    def getSplit(self):
        return self.split