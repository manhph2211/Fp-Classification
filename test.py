from model import AlexNet
import numpy as np 
from dataloader import makeDataSet
import torch
import torchvision.utils
import torch.nn.functional as F
import matplotib.pyplot as plt  



def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)        
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)    
    acc = torch.round(acc) * 100    
    return acc

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train_dataloader,val_dataloader=makeDataSet()

dataiter = iter(val_dataloader)
images, labels = dataiter.next()
print(labels)
classes = ('congphuong', 'messi', 'pique', 'son', 'vantoan')

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

model = AlexNet(5)
model.load_state_dict(torch.load('./football_player.pt'))


outputs = model(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in val_dataloader:
        images, labels = data

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))