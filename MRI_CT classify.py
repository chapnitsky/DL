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


# drive.mount('/content/drive')
# path = '/content/drive/MyDrive/DL/EX4/MRI_CT_data/'
path = f'{os.getcwd()}/MRI_CT_data/'

# !unzip '/content/drive/MyDrive/DL/EX4/MRI_CT_data.zip'
# !pwd
# !ls
# !mkdir MRI_CT_data/all
# !mkdir MRI_CT_data/all/CT
# !mkdir MRI_CT_data/all/MRI
# !cp MRI_CT_data/train/ct/* MRI_CT_data/all/CT
# !cp MRI_CT_data/train/mri/* MRI_CT_data/all/MRI
# !cp MRI_CT_data/test/mri/* MRI_CT_data/all/MRI
# !cp MRI_CT_data/test/ct/* MRI_CT_data/all/CT
# !ls MRI_CT_data/all/MRI/ | wc -l
# !ls MRI_CT_data/all/CT/ | wc -l

training_folder_name = './MRI_CT_data/all'

# New location for the resized images
train_folder = './DATA_OUT/'


# Create resized copies of all of the source images
size = (128,128)
# Create the output folder if it doesn't already exist
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)


def resize_image(src_image, size=(128, 128), bg_color="white"):
    from PIL import Image, ImageOps

    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    return new_image


for root, folders, files in os.walk(training_folder_name):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        # Create a  subfolder in the output location
        saveFolder = os.path.join(train_folder, sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        # Loop through  files in the subfolder (Open each  & resize & save
        file_names = os.listdir(os.path.join(root, sub_folder))
        for file_name in file_names:
            file_path = os.path.join(root, sub_folder, file_name)
            image = Image.open(file_path)
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            resized_image.save(saveAs)


def load_dataset(data_path):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images and transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation)

    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # training data , 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=40,
        num_workers=0,
        shuffle=True
    )

    #  testing data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=40,
        num_workers=0,
        shuffle=True
    )

    return train_loader, test_loader


################################################################
# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset(train_folder)
batch_size = train_loader.batch_size
print("Data loaders ready to read", train_folder)


# Create a neural net class
class Net(nn.Module):
    # Defining the Constructor
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(in_features=18000, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = x.view(-1, 18000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.log_softmax(x, dim=1)


device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"
classes = 2
model = Net(num_classes=classes).to(device)

print(model)

loss_criteria = nn.CrossEntropyLoss()


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


def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
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
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
            # Calculate the average loss and total accuracy for this epoch
            avg_loss = test_loss / batch_count
            print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                avg_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


# Train over 10 epochs (We restrict to 10 for time issues)
model = Net().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
Device = "cuda"
epochs = 10
epoch_nums = []
training_loss = []
validation_loss = []
print('Training on', device)
for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

plt.figure(figsize=(15, 15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
