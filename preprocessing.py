import os
import cv2
import torch
from torch import nn
from sklearn.preprocessing import scale,StandardScaler


IMAGE_SHAPE = 128
# 1. Dataset
DATA_PATH = './data'
player_image_paths = {}
for player_folder in os.listdir(DATA_PATH):
    current_folder = os.path.join(DATA_PATH, player_folder)
    for image_name in os.listdir(current_folder):
        if player_folder not in player_image_paths:
            player_image_paths[player_folder] = [os.path.join(current_folder, image_name)]
        else:
            player_image_paths[player_folder].append(os.path.join(current_folder, image_name))
#print(player_image_paths)
# X_train, y_train, X_val, y_val, X_test, y_test
X = []
y = []
for i, (k, v) in enumerate(player_image_paths.items()):
    # k: messi, pique...
    for image_path in v:
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_SHAPE, IMAGE_SHAPE))
            X.append(image)
            y.append(i)
        except:
            print('Ignore image:', image_path)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

X = torch.reshape(X, [X.shape[0], -1])

# scaler = StandardScaler()
# X= scaler.fit_transform(X.numpy())
# X=torch.from_numpy(X)

# print(X)

torch.save(X, 'X.pt')
torch.save(y, 'y.pt')
