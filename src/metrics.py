from sklearn import metrics as skmetrics

class ClassificationMetrics:

    def __init__(self):

        self.metrics = {'accuracy':self._accuracy,
                        'f1':self._f1,
                        'recall':self._recall,
                        'precision':self._precision,
                        'logloss':self._logloss,
                        'auc':self._auc}


    def __call__(self,metric,y_true,y_pred,y_proba):
        if metric not in self.metrics:
            raise Exception('Metric Not Implemented')

        if metric == 'auc':
            if y_proba is None:
                raise Exception('y proba can not be none in case of auc')
            else return self._auc(y_true=y_true,y_proba=y_proba)

        else:   
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    


    @staticmethod
    def _accuracy(y_true,y_pred):
        return skmetrics.accuracy_score(y_true = y_true, y_pred = y_pred)

   @staticmethod
    def _recall(y_true,y_pred):
        return skmetrics.recall_score(y_true = y_true, y_pred = y_pred)

   @staticmethod
    def _precision(y_true,y_pred):
        return skmetrics.precision_score(y_true = y_true, y_pred = y_pred)

   @staticmethod
    def _logloss(y_true,y_pred):
        return skmetrics.logloss_score(y_true = y_true, y_pred = y_pred)

   @staticmethod
    def _auc(y_true,y_pred):
        return skmetrics.auc_score(y_true = y_true, y_pred = y_pred)

   @staticmethod
    def _f1(y_true,y_pred):
        return skmetrics.f1_score(y_true = y_true, y_pred = y_pred)
