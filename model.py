import torch
import torch.nn as nn
import torch.nn.functional as F


nclasses = 20 

class Classifier(nn.Module):
    def __init__(self, in_features):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, 320)
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, nclasses)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)