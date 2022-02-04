import torch
import torchvision
import torchvision.transforms as trans
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import re
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

classes = {'palm': 0, 'l': 1, 'fist': 2, 'fist_moved': 3, 'thumb': 4, 'palm_moved': 5, 'c': 6,
           'down': 7}


class handsDataSet(Dataset):
    def __init__(self, transformer, data_frame):
        self.data = data_frame
        self.transform = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data['img_id'][index]
        typ = self.data['label'][index]
        sub_dir = typ
        if typ >= 5:
            sub_dir += 2
        _indexes = [m.start() for m in re.finditer('_', img_id)]
        person = int(img_id[_indexes[0] + 1: _indexes[1]])
        path = f'{os.getcwd()}/hands/{person}'
        dir = os.listdir(path)[sub_dir]
        path += f'/{dir}/{img_id}'

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(typ, dtype=torch.long)

        return img, label


class Net(nn.Module):
    # Defining the Constructor
    def __init__(self, num_classes=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3))

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)

        self.fc1 = nn.Linear(in_features=18000, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 18000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.log_softmax(x, dim=1)


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

                    _indexes = [m.start() for m in re.finditer('_', png)]
                    label = int(png[_indexes[1] + 1: _indexes[2]]) - 1
                    if label in [5, 6]:  # Pass the index finger and ok gesture due to vertical & horizontal rotation via transformer
                        break

                    if label not in [0, 1, 2, 3, 4]:  # Reconstruct labels types
                        label -= 2

                    labels.append(label)
                    pngs.append(png)

    data['img_id'] = pngs
    data['label'] = labels
    data.to_csv('./data.csv', index=False)
    return data


def train(model, device, train_loader, optimizer, epoch):
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
        loss = loss_criteria(output, target)
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


def test(model, device, test_loader, n_classes):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            # Calculate the accuracy for this batch

            # make predictions
            _, predicted = torch.max(output.data, 1)
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            correct += torch.sum(target == predicted).item()

            # Calculate the average loss and total accuracy for this epoch
            avg_loss = test_loss / batch_count
            print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                avg_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    print(confusion_matrix)
    arr = confusion_matrix.cpu().detach().numpy()
    fig, ax = plt.subplots()
    ax.matshow(arr)
    plt.ylabel('Real Class')
    plt.xlabel('Predicted Class')
    for (i, j), z in np.ndenumerate(arr):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')
    plt.show()

    print(confusion_matrix.diag() / confusion_matrix.sum(1))

    return avg_loss


if __name__ == "__main__":
    size = (128, 128)
    data_pd = load_data()
    data_transforms = trans.Compose([
        trans.Resize(size),
        # Random Horizontal Flip
        trans.RandomHorizontalFlip(0.5),
        # Random vertical flip
        trans.RandomVerticalFlip(0.3),
        # transform to tensors
        trans.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = 8
    model = Net(num_classes=classes).to(device)
    # model.load_state_dict(torch.load('./NetPerfomance/75_model.pt'))
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10
    epoch_nums = []
    training_loss = []
    test_los = []
    print('Training on', device)
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader, classes)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        test_los.append(test_loss)

    torch.save(model.state_dict(), 'gesture_model.pt')
    model.eval()

    plt.figure(figsize=(15, 15))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, test_los)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'test'], loc='upper right')
    plt.show()
