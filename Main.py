"""
Author: Georgios Voulgaris
Date: 22/12/2020
Description: Test platform to validate various Deep architectures using SSRP dataset. The dataset is comprised of aerial
            images from Ghaziabad India.
            Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and
            validation: 20%), and test: 20% data.
            Architectures used: 3, 4, and 5 CNN, AlexNet, VGG_16.
            Tested images: compressed 50x50, 100x100, and 226x226 pixel images. Note that 50x50 was too small for the
            5 CNNs.
            Test Procedure: 5 runs for each architecture for each of the compressed data. That is 5x50x50 for each
            architecture. Then the Interquartile range, using the median was plotted.
            Plots: Average GPU usage per architecture, Interquartile, and for each architecture an F1 Score heatmap
            for each class.
"""
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Grayscale, Resize
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
from pandas import DataFrame
import shutil
import wandb
wandb.init(project="ssrp")
from models.Deep_Architectures import CNN_3_Net, CNN_4_Net, CNN_5_Net, CNN_5_Net_2, AlexNet, VGG16
from DataPath.ImagesAndFilepaths import ImagesAndFilepaths
import pandas as pd
from torch.utils.data import Subset


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=102)
    parser.add_argument("--train-batch-size", type=int, default=100)
    parser.add_argument("--val-batch-size", type=int, default=100)
    parser.add_argument("--pred-batch-size", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--architecture", type=int, default=1, help="1. 3_CNN, 2. 4_CNN, 3. 5_CNN, 4. AlexNet, "
                                                                    "5. VGG16, 6. 5_CNN_2")
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--dataset", type=int, default=1, help="1. Training_Data_2018_2014, "
                                                               "2. VAE_Training_Data_2018_2014")

    return parser.parse_args()


def step(x, y, net, optimizer, loss_function, train):

    with torch.set_grad_enabled(train):
        outputs = net(x)
        acc = outputs.argmax(dim=1).eq(y).sum().item()
        loss = loss_function(outputs, y)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = []
    for x, _, _ in loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).cpu()
    return all_preds


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, net, optimizer):
    print("=> loading checkpoint...")
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def main():

    args = arguments()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    transforms = Compose([Resize((args.height, args.width)), ToTensor()])
    if args.dataset == 1:
        dataset = ImageFolder("Training_Data_2018_2014", transform=transforms)
        print("Dataset: Training_Data_2018_2014")
    else:
        dataset = ImageFolder("VAE_Training_Data_2018_2014", transform=transforms)
        print("Dataset: VAE_Training_Data_2018_2014")

    LABELS = dataset.classes
    input_size = dataset[0][0].shape
    y1 = dataset.targets
    dataset_len = len(dataset)

    trainval_idx, test_idx, y_trainval_idx, y_test_idx = train_test_split(np.arange(dataset_len), y1, test_size=0.2,
                                                                          stratify=y1, random_state=args.random_state,
                                                                          shuffle=True)
    X2 = trainval_idx
    y2 = y_trainval_idx
    train_idx, val_idx, y_train_idx, y_val_idx = train_test_split(X2, y2, test_size=0.2, stratify=y2,
                                                                  random_state=args.random_state, shuffle=True)

    train_without_filepaths = Subset(dataset, train_idx)
    val_without_filepaths = Subset(dataset, val_idx)
    test_without_filepaths = Subset(dataset, test_idx)
    filepaths = np.array(tuple(zip(*dataset.imgs))[0])
    train = ImagesAndFilepaths(train_without_filepaths, filepaths[train_idx])
    val = ImagesAndFilepaths(val_without_filepaths, filepaths[val_idx])
    test = ImagesAndFilepaths(test_without_filepaths, filepaths[test_idx])
    train_loader = DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.val_batch_size, shuffle=False)
    prediction_loader = DataLoader(test, batch_size=args.pred_batch_size)

    if args.architecture == 1:
        net = CNN_3_Net(input_size).to(device)
    elif args.architecture == 2:
        net = CNN_4_Net(input_size).to(device)
    elif args.architecture == 3:
        net = CNN_5_Net(input_size).to(device)
    elif args.architecture == 4:
        net = AlexNet(input_size).to(device)
    elif args.architecture == 5:
        net = VGG16(input_size).to(device)
    else:
        net = CNN_5_Net_2(input_size).to(device)

    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    if args.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), net, optimizer)

    # with open("CNN_model.log", "a") as f:
    for epoch in range(args.epochs):
        net.train()
        sum_acc = 0
        for x, y, filepath in train_loader:
            x = x.to(device)
            y = y.to(device)
            acc, loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += acc
        train_avg_acc = sum_acc / len(train_loader)

        net.eval()
        sum_acc = 0

        if epoch % 10 == 0:
            checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

        for x, y, filepath in val_loader:
            x = x.to(device)
            y = y.to(device)
            val_acc, val_loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)

        print(f"Epoch: {epoch} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)

    train_preds = get_all_preds(net, loader=prediction_loader, device=device)
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"The label the network predicts strongly: {train_preds.argmax(dim=1)}")
    predictions = train_preds.argmax(dim=1)

    # Write on a csv file the image ids, target and predicted values
    signal_dict = {
        'Filename': filepaths[test_idx],
        'True Value': y_test_idx,
        'Prediction': predictions.tolist(),
    }
    df = pd.DataFrame(signal_dict)
    df.to_csv('Predictions.csv', index=False)

    plt.figure(figsize=(10, 10))
    # Confusion Matrix
    wandb.sklearn.plot_confusion_matrix(y_test_idx, train_preds.argmax(dim=1), LABELS)
    # Class proportions
    wandb.sklearn.plot_class_proportions(y_train_idx, y_test_idx, LABELS)
    precision, recall, f1_score, support = score(y_test_idx, train_preds.argmax(dim=1))
    test_acc = accuracy_score(y_test_idx, train_preds.argmax(dim=1))

    print(f"Test Accuracy: {test_acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")
    print(f"support: {support}")

    # Test data saved in Excel document
    df = DataFrame({'Test Accuracy': test_acc, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                    'support': support})
    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    df.to_csv('test.csv', index=False)
    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)

    wandb.save('test.csv')
    wandb.save('my_checkpoint.pth.tar')
    wandb.save('Predictions.csv')


if __name__ == "__main__":
    main()