from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))

# We convert every value below 0.5 to 0. Otherwise, we set it to 1
def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true) 
