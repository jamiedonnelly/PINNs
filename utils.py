import torch 
from torch import nn
from losses import RMSELoss

class EarlyStopping():
    
    def __init__(self,metric,tolerance,patience):
        self.metric = metric
        self.tolerance = tolerance
        self.patience = patience
        self._scores = torch.tensor([])
        self._loss = self._metric_check()
        
    def _metric_check(self):
        if self.metric == "mse":
            return nn.MSELoss()
        if self.metric == "rmse":
            return RMSELoss()
            
    def _check_abs(self,index):
        if torch.abs(self._scores[index]-self._scores[index-1]) < self.tolerance: 
            return True
        else:
            return False
            
    def _check_increase(self,index):
        if self._scores[index] > self._scores[index-1]:
            return True
        else:
            return False

    def _evaluate_convergence(self):
        return torch.all(torch.tensor([self._check_abs(i) for i in range(int(-1*self.patience),0)]))

    def _evaluate_increase(self):
        return torch.all(torch.tensor([self._check_increase(i) for i in range(int(-1*self.patience),0)]))        

    def _evaluate_predictions(self,model,Xtest,ytest):
        predictions = model(Xtest)
        score = self._loss(predictions,ytest)
        self._scores = torch.hstack([self._scores,score])
        print(f"Test {len(self._scores)+1} Evaluation Score: {score.item():.4f}")
            
    def __call__(self,model,Xtest,ytest):
        self._evaluate_predictions(model,Xtest,ytest)
        if self._scores.shape[0] > self.patience:
            if self._evaluate_convergence() or self._evaluate_increase():
                raise ValueError
