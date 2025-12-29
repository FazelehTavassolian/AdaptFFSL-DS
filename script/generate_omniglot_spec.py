from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import cv2

n_episodes = 1000

data_root = Path('/home/jericho/Project/federated-meta-learning/Data/fashion_mnist')
train_x = data_root.joinpath('fashion-mnist_train.csv')
valid_x = data_root.joinpath('fashion-mnist_test.csv')

save_path = Path('/home/jericho/Project/federated-meta-learning/Data/fashion_mnist-fs')
save_path.mkdir(parents=True, exist_ok=True)

# Create Raw
step = 0
raw_save = save_path.joinpath('raw')
raw_save.mkdir(parents=True, exist_ok=True)
for idx, data in pd.read_csv(train_x).iterrows():
    label = data[0]
    im_path = raw_save.joinpath(str(label))
    im_path.mkdir(parents=True, exist_ok=True)

    pixels = np.array(data[1:]).reshape((28, 28,1))
    pixels = np.concatenate([pixels,pixels,pixels],axis=-1)
    cv2.imwrite(str(im_path.joinpath(f'IMG-{step}.jpg')),pixels)
    step += 1

for idx, data in pd.read_csv(valid_x).iterrows():
    label = data[0]
    im_path = raw_save.joinpath(str(label))
    im_path.mkdir(parents=True, exist_ok=True)

    pixels = np.array(data[1:]).reshape((28, 28,1))
    pixels = np.concatenate([pixels,pixels,pixels],axis=-1)
    cv2.imwrite(str(im_path.joinpath(f'IMG-{step}.jpg')),pixels)
    step += 1

# save val
vl_save = save_path.joinpath('val')
vl_save.mkdir(parents=True, exist_ok=True)
# for src_f in valid.glob("*/*/*"):
#     dest_fol = f'{src_f.parent.parent.stem}-{src_f.parent.stem}'
#     dest_save = vl_save.joinpath(dest_fol)
#     dest_save.mkdir(parents=True, exist_ok=True)
#     shutil.copy(src_f, dest_save.joinpath(src_f.name))

tr_save = save_path.joinpath('train')
tr_save.mkdir(parents=True, exist_ok=True)
# way5_shot1 = []
# way5_shot5 = []
# for ep_idx in range(n_episodes):
#     all_parents = list(vl_save.glob("*"))
#     # 5 Way 1 shot
#     way_idx = np.random.choice(all_parents, 5, replace=False)
#     select = {'Support': [], "Query": []}
#     for src_f in way_idx:
#         files = list(src_f.glob('*'))
#         f_names = []
#         for file in files[:1]:
#             f_names.append(f'{file.parent.stem}/{file.name}')
#         select['Support'].append(f_names)
#         f_names = []
#         end = min(len(files) - 1, 15)
#         for file in files[1:end]:
#             f_names.append(f'{file.parent.stem}/{file.name}')
#         select['Query'].append(f_names)
#     way5_shot1.append(select)
#
#     # 5 Way 5 Shot
#     way_idx = np.random.choice(all_parents, 5, replace=False)
#     select = {'Support': [], "Query": []}
#     for src_f in way_idx:
#         files = list(src_f.glob('*'))
#         f_names = []
#         for file in files[:5]:
#             f_names.append(f'{file.parent.stem}/{file.name}')
#         select['Support'].append(f_names)
#         f_names = []
#         for file in files[5:]:
#             f_names.append(f'{file.parent.stem}/{file.name}')
#         select['Query'].append(f_names)
#     way5_shot5.append(select)
#
# with open(str(save_path.joinpath('val1000Episode_5_way_1_shot.json')), 'w') as f:
#     json.dump(way5_shot1, f)
#
# with open(str(save_path.joinpath('val1000Episode_5_way_5_shot.json')), 'w') as f:
#     json.dump(way5_shot5, f)
