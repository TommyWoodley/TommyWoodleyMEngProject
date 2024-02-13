import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.dropout3 = nn.Dropout(p=0.6)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        return self.fc4(x)
    
class Agent(QNetwork):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
        super().__init__(state_size, action_size, seed, fc1_units, fc2_units, fc3_units)