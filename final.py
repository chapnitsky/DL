import itertools
import sys
import os
import pandas as pd
import os
import numpy
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transformers
import torch.nn.functional as F
from PIL import Image
import pickle
from scipy import ndimage
from sklearn.utils import shuffle
import time
import math
import re

classes = {'palm': 0, 'l': 1, 'fist': 2, 'fist_moved': 3, 'thumb': 4, 'index': 5, 'ok': 6, 'palm_moved': 7, 'c': 8,
           'down': 9}


class handsDataSet(Dataset):
    def __init__(self, transformer, data_frame):
        self.data = data_frame
        self.transform = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data['img_id'][index]
        typ = self.data['label'][index]

        _indexes = [m.start() for m in re.finditer('_', img_id)]
        person = int(img_id[_indexes[0] + 1: _indexes[1]])
        path = f'{os.getcwd()}/hands/{person}'
        dir = os.listdir(path)[typ]
        path += f'/{dir}/{img_id}'

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(float(typ))

        return img, label


class HandGestureNet(torch.nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data

    Summary
    -------
        Deep Learning Model for Hand Gesture classification using pose data only (no need for RGBD)
        The model computes a succession of [convolutions and pooling] over time independently on each of the 66 (= 22 * 3) sequence channels.
        Each of these computations are actually done at two different resolutions, that are later merged by concatenation
        with the (pooled) original sequence channel.
        Finally, a multi-layer perceptron merges all of the processed channels and outputs a classification.

    TL;DR:
    ------
        input ------------------------------------------------> split into n_channels channels [channel_i]
            channel_i ----------------------------------------> 3x [conv/pool/dropout] low_resolution_i
            channel_i ----------------------------------------> 3x [conv/pool/dropout] high_resolution_i
            channel_i ----------------------------------------> pooled_i
            low_resolution_i, high_resolution_i, pooled_i ----> output_channel_i
        MLP(n_channels x [output_channel_i]) -------------------------> classification

    Article / PDF:
    --------------
        https://ieeexplore.ieee.org/document/8373818

    Please cite:
    ------------
        @inproceedings{devineau2018deep,
            title={Deep learning for hand gesture recognition on skeletal data},
            author={Devineau, Guillaume and Moutarde, Fabien and Xi, Wang and Yang, Jie},
            booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
            pages={106--113},
            year={2018},
            organization={IEEE}
        }
    """

    def __init__(self, n_channels=3, n_classes=10):

        super(HandGestureNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layers ----------------------------------------------
        self.all_conv_high = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 7), padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(7, 7), padding=3),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(7, 7), padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ) for joint in range(n_channels)])

        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ) for joint in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.MaxPool2d(2),
            torch.nn.MaxPool2d(2)
        ) for joint in range(n_channels)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * n_channels * 12, out_features=1936),
            # <-- 12: depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1936, out_features=n_classes)
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.all_conv_high, self.all_conv_low, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv2d":
                    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        Arguments
        ---------
            input: a tensor of gestures of shape (batch_size, duration, n_channels)
                   (where n_channels = 3 * n_joints for 3D pose data)
        """

        # Work on each channel separately
        all_features = []

        for channel in range(0, self.n_channels):
            input_channel = input[:, :, channel]

            # Add a dummy (spatial) dimension for the time convolutions
            # Conv1D format : (batch_size, n_feature_maps, duration)
            input_channel = input_channel.unsqueeze(3)

            high = self.all_conv_high[channel](input_channel)
            low = self.all_conv_low[channel](input_channel)
            ap_residual = self.all_residual[channel](input_channel)

            # Time convolutions are concatenated along the feature maps axis
            output_channel = torch.cat([
                high,
                low,
                ap_residual
            ], dim=1)
            all_features.append(output_channel)

        # Concatenate along the feature maps axis
        all_features = torch.cat(all_features, dim=1)

        # Flatten for the Linear layers
        all_features = all_features.view(-1,
                                         9 * self.n_channels * 12)  # <-- 12: depends of the initial sequence length (100).
        # If you have shorter/longer sequences, you probably do NOT even need to modify the modify the network architecture:
        # resampling your input gesture from T timesteps to 100 timesteps will (surprisingly) probably actually work as well!

        # Fully-Connected Layers
        output = self.fc(all_features)

        return output


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


def resize_sequences_length(x_train, x_test, final_length=100):
    """
    Resize the time series by interpolating them to the same length
    """
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    x_train = numpy.array([numpy.array(
        [ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))]).T for x_i
                           in x_train])
    x_test = numpy.array([numpy.array(
        [ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))]).T for x_i
                          in x_test])
    return x_train, x_test


def shuffle_dataset(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """Shuffle the train/test data consistently."""
    # note: add random_state=0 for reproducibility
    x_train, y_train_14, y_train_28 = shuffle(x_train, y_train_14, y_train_28)
    x_test, y_test_14, y_test_28 = shuffle(x_test, y_test_14, y_test_28)
    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def preprocess_data(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    """
    Preprocess the data as you want: update as you want!
        - possible improvement idea: make a PCA here
    """
    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = shuffle_dataset(x_train, x_test, y_train_14,
                                                                                    y_train_28, y_test_14, y_test_28)
    x_train, x_test = resize_sequences_length(x_train, x_test, final_length=100)
    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def convert_to_pytorch_tensors(x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28):
    # as numpy
    y_train_14, y_train_28, y_test_14, y_test_28 = numpy.array(y_train_14), numpy.array(y_train_28), numpy.array(
        y_test_14), numpy.array(y_test_28)

    # -- REQUIRED by the pytorch loss function implementation --
    # Remove 1 to all classes items (1-14 => 0-13 and 1-28 => 0-27)
    y_train_14, y_train_28, y_test_14, y_test_28 = y_train_14 - 1, y_train_28 - 1, y_test_14 - 1, y_test_28 - 1

    # as torch
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train_14, y_train_28, y_test_14, y_test_28 = torch.from_numpy(y_train_14), torch.from_numpy(
        y_train_28), torch.from_numpy(y_test_14), torch.from_numpy(y_test_28)

    # -- REQUIRED by the pytorch loss function implementation --
    # correct the data type (for the loss function used)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train_14, y_train_28, y_test_14, y_test_28 = y_train_14.type(torch.LongTensor), y_train_28.type(
        torch.LongTensor), y_test_14.type(torch.LongTensor), y_test_28.type(torch.LongTensor)

    return x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28


def batch(tensor, batch_size=40):
    """Return a list of (mini) batches"""
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i + 1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i + 1) * batch_size])
        i += 1


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:02d}m {:02d}s'.format(int(m), int(s))


def get_accuracy(model, x, y_ref):
    """Get the accuracy of the pytorch model on a batch"""
    acc = 0.
    model.eval()
    with torch.no_grad():
        predicted = model(x)
        _, predicted = predicted.max(dim=1)
        acc = 1.0 * (predicted == y_ref).sum().item() / y_ref.shape[0]

    return acc


#
# def train(model, criterion, optimizer,
#           x_train, y_train, x_test, y_test,
#           force_cpu=False, num_epochs=5):
#     # use a GPU (for speed) if you have one
#     device = torch.device("cuda") if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
#     model = model.to(device)
#     x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)
#
#     # (bonus) log accuracy values to visualize them in tensorboard:
#
#     # Prepare all mini-batches
#     x_train_batches = batch(x_train)
#     y_train_batches = batch(y_train)
#
#     # Training starting time
#     start = time.time()
#
#     print('[INFO] Started to train the model.')
#     print('Training the model on {}.'.format('GPU' if device == torch.device('cuda') else 'CPU'))
#
#     for ep in range(num_epochs):
#
#         # Ensure we're still in training mode
#         model.train()
#
#         current_loss = 0.0
#
#         for idx_batch, train_batches in enumerate(zip(x_train_batches, y_train_batches)):
#             # get a mini-batch of sequences
#             x_train_batch, y_train_batch = train_batches
#
#             # zero the gradient parameters
#             optimizer.zero_grad()
#
#             # forward
#             outputs = model(x_train_batch)
#
#             # backward + optimize
#             # backward
#             loss = criterion(outputs, y_train_batch)
#             loss.backward()
#             # optimize
#             optimizer.step()
#             # for an easy access
#             current_loss += loss.item()
#
#         train_acc = get_accuracy(model, x_train, y_train)
#         test_acc = get_accuracy(model, x_test, y_test)
#
#         print(
#             'Epoch #{:03d} | Time elapsed : {} | Loss : {:.4e} | Accuracy_train : {:.4e} | Accuracy_test : {:.4e}'.format(
#                 ep + 1, time_since(start), current_loss, train_acc, test_acc))
#
#     print('[INFO] Finished training the model. Total time : {}.'.format(time_since(start)))
#

def train(model, device, train_loader, optimizer, criterion, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the optimizer
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = criterion(output, target)
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


if __name__ == '__main__':
    # Load the dataset
    data_pd = load_data()

    data_transforms = transformers.Compose([
        # Random Horizontal Flip
        transformers.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transformers.RandomVerticalFlip(0.3),
        # transform to tensors
        transformers.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transformers.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_set = handsDataSet(data_frame=data_pd, transformer=data_transforms)

    TRAIN_PERCENT = 0.8
    total_len = len(data_set)
    train_size = int(TRAIN_PERCENT * total_len)
    test_size = int(total_len - train_size)

    x_train, x_test = random_split(data_set, [train_size, test_size])

    # training data , 40-image batches
    train_loader = torch.utils.data.DataLoader(
        x_train,
        batch_size=40,
        num_workers=0,
        shuffle=True
    )

    #  testing data
    test_loader = torch.utils.data.DataLoader(
        x_test,
        batch_size=40,
        num_workers=0,
        shuffle=True
    )
    ##################################################################################
    # y_train_14, y_train_28, y_test_14, y_test_28 = 0, 0, 0, 0

    # x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = preprocess_data(x_train, x_test, y_train_14,
    #                                                                                 y_train_28,
    #                                                                                 y_test_14, y_test_28)

    # # Convert to pytorch variables
    # x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = convert_to_pytorch_tensors(x_train, x_test,
    #                                                                                            y_train_14,
    #                                                                                            y_train_28, y_test_14,
    #                                                                                            y_test_28)
    ####################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HandGestureNet(n_channels=3, n_classes=10).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    num_epochs = 20

    train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, device=device,
          epoch=num_epochs)

    torch.save(model.state_dict(), 'gesture_pretrained_model.pt')

    model.load_state_dict(torch.load('gesture_pretrained_model.pt'))
    model.eval()

    # make predictions
    with torch.no_grad():
        demo_gesture_batch = torch.randn(32, 100, 66)
        predictions = model(demo_gesture_batch)
        _, predictions = predictions.max(dim=1)
        print("Predicted gesture classes: {}".format(predictions.tolist()))
