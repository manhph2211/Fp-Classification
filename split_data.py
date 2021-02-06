import os
import os.path as osp
import random
import shutil



ROOT_DATA='./data'
soccer_name = os.listdir(ROOT_DATA)
# print(soccer_name)

soccer_paths = {}
for sc in soccer_name:
    soccer_paths[sc] = []

for sc in soccer_name:
    folder_path = osp.join(ROOT_DATA, sc)
    image_names = os.listdir(folder_path)
    soccer_paths[sc] = [osp.join(folder_path, _name) for _name in image_names]


DEST_FOL = "./data_after_splitting"
TRAIN_FOL = "./data_after_splitting/train"
VAL_FOL = "./data_after_splitting/val"

for sc, sc_paths in soccer_paths.items():
    sc_train_fol = osp.join(TRAIN_FOL, sc)
    sc_val_fol = osp.join(VAL_FOL, sc)
    
    #print(sc)
    if not osp.exists(sc_train_fol):
        os.makedirs(sc_train_fol)
    if not osp.exists(sc_val_fol):
        os.makedirs(sc_val_fol)
    
    random.shuffle(sc_paths)
    train_idx = int(0.6*len(sc_paths))
    
    for path in sc_paths[:train_idx]:
        shutil.copy2(path, sc_train_fol)
    
    for path in sc_paths[train_idx:]:
        shutil.copy2(path, sc_val_fol)





