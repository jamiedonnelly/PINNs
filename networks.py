import torch 
from torch import nn
from torch.functional import F

# Define Model 

class CNN(nn.Module):

    def __init__(self,output_dim):
        super(CNN,self).__init__()
        # Conv1 and BN
        self.conv1 = nn.Conv1d(1,32,1)
        self.conv1_bn = nn.BatchNorm1d(32)
        # Conv2 and BN
        self.conv2 = nn.Conv1d(32,64,1)
        self.conv2_bn = nn.BatchNorm1d(64)
        # Conv3 and BN
        self.conv3 = nn.Conv1d(64,128,1)
        self.conv3_bn = nn.BatchNorm1d(128)
        # Conv4 and BN
        self.conv4 = nn.Conv1d(128,256,1)
        self.conv4_bn = nn.BatchNorm1d(256)
        # FC1 and BN
        self.fc1 = nn.Linear(256*9,1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        # FC2 and BN
        self.fc2 = nn.Linear(1024,2048)
        self.fc2_bn = nn.BatchNorm1d(2048)
        # FC3 and BN
        self.fc3 = nn.Linear(2048,4096)
        self.fc3_bn = nn.BatchNorm1d(4096)
        # FC4 and BN
        self.fc4 = nn.Linear(4096,1024)
        self.fc4_bn = nn.BatchNorm1d(1024)
        # Final FC
        self.fc5 = nn.Linear(1024,output_dim)
        # Dropout 
        self.dropout = nn.Dropout(0.50)

    def forward(self,x):
        # Conv1 
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        # Conv2
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        # Conv3
        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        # Conv4
        x = self.conv4(x)
        x = F.relu(self.conv4_bn(x))
        
        # Flatten
        x = torch.flatten(x,1)
        
        # Fc1
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        x = self.dropout(x)
        # Fc2
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.dropout(x)
        # Fc3
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        x = self.dropout(x)
        # Fc4
        x = self.fc4(x)
        x = F.relu(self.fc4_bn(x))
        x = self.dropout(x)
        # Fc5
        x = self.fc5(x)
        return x
