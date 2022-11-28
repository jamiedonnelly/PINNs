import torch 
from torch import nn
from losses import RMSELoss

class EarlyStopping():
    
    def __init__(self,metric,tolerance,patience):
        self.metric = metric
        self.tolerance = tolerance
        self.patience = patience
        self._scores = torch.tensor([])
        self.bool = True
        self._eval = self._metric_check()
        
    def _metric_check(self):
        if self.metric == "mse":
            return nn.MSELoss()
        if self.metric == "rmse":
            return RMSELoss()
            
    def _check(self,index):
        if torch.abs(self._scores[index]-self._scores[index-1]) < self.tolerance: 
            return True
        else:
            return False
            
    def _evaluate_bool(self):
        return torch.all(torch.tensor([self._check(i) for i in range(int(-1*self.patience),0)]))
            
    def __call__(self,model,Xtest,ytest):
        if self.bool:
            predictions = model(Xtest)
            score = self._eval(predictions,ytest)
            print(f"Test {len(self._scores)+1} Evaluation Score: {score.item()}")
            self._scores = torch.hstack([self._scores,score])
            if self._scores.shape[0] > self.patience:
                if self._evaluate_bool():
                    self.bool = False
        else:
            raise ValueError