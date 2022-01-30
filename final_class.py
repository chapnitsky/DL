import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns
# %matplotlib inline
# from google.colab import drive
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image

training_folder_name = './MRI_CT_data/all'

# New location for the resized images
train_folder = './DATA_OUT/'


# Create resized copies of all of the source images
size = (128,128)
# Create the output folder if it doesn't already exist
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)


def load_data(folder_path='./hands', shap=(640, 240)):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    data = pd.DataFrame(columns=['img_id', 'label'])
    persons = 10
    pngs = []
    labels = []
    for i in range(persons):
        sub_dirs = os.listdir(f'{folder_path}/{i}')
        for dir in sub_dirs:
            path = f'{folder_path}/{i}/{dir}'
            subs = os.listdir(path)
            for png in subs:
                if '.png' in png:
                    png = str(png)
                    pngs.append(png)

                    _indexes = [m.start() for m in re.finditer('_', png)]
                    label = int(png[_indexes[1] + 1: _indexes[2]]) - 1
                    labels.append(label)

    data['img_id'] = pngs
    data['label'] = labels
    data.to_csv('./data.csv', index=False)
    return data
