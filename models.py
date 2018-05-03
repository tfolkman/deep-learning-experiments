import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Our network takes in an image and tries to predict the quality
    of taking each of our 9 actions given that state (the image)
    """

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.bn = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2)
        self.head = nn.Linear(32*18*38, n_actions)

    def forward(self, x):
        x = self.maxpool(self.bn(F.elu(self.conv1(x))))
        return self.head(x.view(x.size(0), -1))

 
class NNPolicy(nn.Module):
    def __init__(self, n_inputs=4, n_hidden=4, n_outputs=2):
        super(NNPolicy, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        self.hidden1 = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        
    def forward(self, x):
        output = F.elu(self.hidden1(x))
        output = self.out(output)
        return output.view(x.size(0),-1)