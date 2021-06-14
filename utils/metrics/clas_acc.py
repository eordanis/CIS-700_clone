from sklearn.metrics import accuracy_score
from utils.metrics.Metrics import Metrics

class ACC(Metrics):
    def __init__(self, if_use=True, gpu=False):
        super().__init__()
        self.name = 'acc'

        self.if_use = if_use
        self.model = None
        self.data_loader = None
        self.gpu = gpu
        self.predictions = []
        self.y = []
        

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def reset(self, predictions, y):
        self.predictions = predictions
        self.y = y

    def get_score(self):
        return self.cal_acc(self.predictions, self.y)

    def cal_acc(self, predictions, y):
        if len(y) != 0 and len(predictions) != 0:
            size = min(len(y), len(predictions))
            if size > 32:
                size = 32
            self.y = y.tolist()[:size]
            self.predictions = predictions[:size]
            yVals = []
            for x in self.y:
                yVals.append(x[0])
            acc =  accuracy_score(self.predictions, yVals)
            print("ACC:")
            print(acc)
            return acc

        