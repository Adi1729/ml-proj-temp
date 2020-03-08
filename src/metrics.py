from sklearn import metrics

class ClassificationMetrics:

    def __init__(self):

        self.metrics = {'accuracy':self.accuracy}

    @staticmethod