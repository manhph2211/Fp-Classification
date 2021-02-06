import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from model import AlexNet
from dataloader import makeDataSet


train_dataloader,val_dataloader=makeDataSet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlexNet(5)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("Begin training.")

for epoch in tqdm(range(1, 100)):    
    # TRAINING    
    running_loss = 0.0

    model.train()
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        torch.save(model,'football_player.pt')
        
    print('Epoch [%d] loss: %.3f' %
            (epoch + 1, running_loss))
