import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ipdb import set_trace

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def fit(epoch, model, trainloader, testloader):
    train_correct = 0
    train_total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to("cuda"), y.to("cuda")
        # 权重参数清零
        optimizer.zero_grad()
        # 前向与反向传播
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            train_correct += (y_pred==y).sum().item()
            train_total += y.size(0)
            running_loss += loss.item()

    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = train_correct / train_total

    test_correct = 0
    test_running_loss = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to("cuda"), y.to("cuda")
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        test_correct += (y_pred==y).sum().item()
        test_running_loss += loss.item()
        test_total += y.size(0)

    test_epoch_acc = test_correct / test_total
    test_epoch_loss = test_running_loss / len(testloader.dataset)

    return  epoch_loss, epoch_acc, test_epoch_acc, test_epoch_loss

if __name__ == "__main__":
    train_dir = os.path.join("metadata/paris_train/")
    test_dir = os.path.join("metadata/paris_test/")
    # set_trace()

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(192, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.2),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_ds = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    test_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    test_ds = torchvision.datasets.ImageFolder(
        train_dir,
        transform=test_transform
    )

    BATCH_SIZE = 16
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
    )

    model = torchvision.models.resnet101(pretrained=False, progress=True)

    for param in model.parameters():
        param.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 4)

    if torch.cuda.is_available():
        model.to("cuda")

    loss_fn = nn.CrossEntropyLoss()

    from torch.optim import lr_scheduler

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_acc = []
    train_loss= []
    test_acc = []
    test_loss = []
    Epochs = 100
    for epoch in range(Epochs):
        epoch_loss, epoch_acc, test_epoch_acc, test_epoch_loss = fit(epoch, model, train_dl, test_dl)

        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        print("epoch:", epoch,
              "train_loss:", round(epoch_loss, 3),
              "train_acc:", round(epoch_acc, 3),
              "test_loss:", round(test_epoch_loss, 3),
              "test_acc:", round(test_epoch_acc, 3)
              )
    plt.plot(range(1, Epochs + 1), train_loss, label="train_loss")
    plt.plot(range(1, Epochs + 1), test_loss, label="test_loss")
    plt.legend()

    plt.plot(range(1, Epochs + 1), train_acc, label="train_acc")
    plt.plot(range(1, Epochs + 1), test_acc, label="test_acc")
    plt.show()
