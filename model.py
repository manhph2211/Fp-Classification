import torch
import torchvision
import torch.nn as nn
import torch.optim as optim



class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class Model(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, out_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden2, hidden2)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(hidden2, hidden2)
        
        self.linear6 = nn.Linear(hidden2, hidden2)
        
        self.linear7 = nn.Linear(hidden2, hidden2)
        #self.linear8 = nn.Linear(hidden2, hidden2)
        self.classifier = nn.Linear(hidden2, out_dim)
        self.softmax=nn.Softmax()
    def forward(self, X):
        X = self.relu1(self.linear1(X))
        X = self.relu2(self.linear2(X))
        X = self.linear3(X)
        X=self.linear4(X)
        #X = self.relu4(self.linear4(X))
        X=self.linear5(X)
        X=self.linear6(X)
        X = self.linear7(X)
        #X = self.linear8(X)
        #X=self.classifier(X)
        #X = self.softmax(self.classifier(X))
        X = self.classifier(X)
        return X