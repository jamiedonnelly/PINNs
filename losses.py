import torch 
from torch import nn
from torch.functional import F

# Physics Informed Loss
class PhysLoss(torch.nn.modules.loss._Loss):
    def __init__(self,inputs,targets,n_bnd,dx,dt,nx,ny,verbose=True):
        super(PhysLoss,self).__init__()
        self.inputs = inputs
        self.targets = targets 
        self.n_bnd = n_bnd
        self.dx = dx
        self.dt = dt
        self.nx = nx
        self.ny = ny
        self.verbose = verbose
        # Initialise latent variables that will be updated during training
        self.data_fit, self.lower_loss, self.upper_loss = None, None, None
    
    def _sum_bct(self,index):
        return self.inputs[index,:,-1].squeeze(1)*self.n_bnd*self.dt
    
    def _vol_calculate(self,value):
        return torch.sum(value,dim=1)*(self.dx**2)
    
    def _mse(self,predicted,index):
        return torch.mean((self.targets[index]-predicted)**2,dim=1)
    
    def _batch_error(self,batch_pred,batch_index):
        data_fit = self._mse(batch_pred,batch_index)
        vt_pred = self._vol_calculate(batch_pred)
        vt_prev = self._vol_calculate(self.targets[batch_index-1])
        vt_next = self._vol_calculate(self.targets[batch_index+1])
        St = self._sum_bct(batch_index)
        St_next = self._sum_bct(batch_index+1)
        return data_fit, F.relu(vt_pred - vt_prev - St)/(self.dx*self.nx*self.ny), F.relu(vt_next - vt_pred - St_next)/(self.dx*self.nx*self.ny)
    
    def __call__(self,batch_pred,batch_index):
        data_fit, c1, c2 = self._batch_error(batch_pred,batch_index)
        self.data_fit, self.lower_loss, self.upper_loss = torch.round(torch.mean(data_fit),decimals=4), torch.round(torch.mean(c1),decimals=4), torch.round(torch.mean(c2),decimals=4)
        batch_loss = data_fit + c1 + c2 
        if not self.verbose:
            return torch.mean(batch_loss)
        else:
            print(f"Data-fit loss: {self.data_fit}")
            print(f"Lower-loss : {self.lower_loss}")
            print(f"Upper-loss : {self.upper_loss}")
            print(f"Full  loss: {torch.mean(batch_loss)}\n")
            return torch.mean(batch_loss)
        
# Torch RMSE Implementation
class RMSELoss(nn.MSELoss):
    def __init__(self):
        super(RMSELoss,self).__init__()
        pass

    def __call__(self, target, predicted):
        loss = torch.sqrt(super().__call__(target,predicted))
        return loss

# Custom RMSE 
def custom_rmse(true,pred):
    maxtrue, maxpred = torch.max(true,dim=0).values, torch.max(pred,dim=0).values
    indexes = []
    for i in range(len(maxtrue)):
        if (maxpred[i]>0) or (maxtrue[i]>0):
            indexes.append(i)
    return torch.sqrt(torch.mean((true[:,indexes]-pred[:,indexes])**2))
