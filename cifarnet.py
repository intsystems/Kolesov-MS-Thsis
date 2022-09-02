from torch import nn
from collections import OrderedDict

class CIFARNet(nn.Module):
    def __init__(self, num_classes=100, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([ # 32
            ('conv1', nn.Conv2d(3, hidden_dim, 7, bias=False)), # 26
            ('bn1', nn.BatchNorm2d(hidden_dim)),
            ('relu1', nn.LeakyReLU()),
            ('maxpool', nn.MaxPool2d(2, 2)), # 13

            ('conv2', nn.Conv2d(hidden_dim, 2*hidden_dim, 5, bias=False)), # 9
            ('bn2', nn.BatchNorm2d(2*hidden_dim)),
            ('relu2', nn.LeakyReLU()),

            ('conv3', nn.Conv2d(2*hidden_dim, 2*hidden_dim, 5, bias=False)), # 5
            ('bn3', nn.BatchNorm2d(2*hidden_dim)),
            ('relu3', nn.LeakyReLU()),

            ('conv4', nn.Conv2d(2*hidden_dim, 4*hidden_dim, 5, bias=False)), # 1
            ('bn4', nn.BatchNorm2d(4*hidden_dim)),
            ('relu4', nn.LeakyReLU()),
            ('flatten', nn.Flatten(1, -1))
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4*hidden_dim, 4*hidden_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(4*hidden_dim)),
            ('relu1', nn.LeakyReLU()),
            ('fc2', nn.Linear(4*hidden_dim, num_classes))
        ]))
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)