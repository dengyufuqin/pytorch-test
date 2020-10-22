import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from model.models import *
from tqdm import tqdm

def accuracy(output, target):
    output = np.array(output.cpu().detach().numpy())
    target = np.array(target.cpu().detach().numpy())
    prec = 0
    for i in range(output.shape[0]):
        pos = np.unravel_index(np.argmax(output[i]), output.shape)
        pre_label = pos[1]
        if pre_label == target[i]:
            prec += 1

    prec /= target.size
    prec *= 100
    return prec


device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = CustomDataset(datatxt='data/train.txt',
                           transform=transform)

test_data = CustomDataset(datatxt='data/test.txt',
                          transform=test_transform)

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

ResNet = ResNet50(ResidualBlock, [3, 4, 6, 3],num_classes=2).to(device)
# ResNet = MobileNetV1().to(device)
# ResNet = MobileNetV2().to(device)
# ResNet = MobileNetV3().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ResNet.parameters(), lr=learning_rate, momentum=0.9)

# for update learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs =    ResNet(images)
        loss = criterion(outputs, labels)

        prec = accuracy(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 100 == 0:
        #     print ("\nEpoch [{}/{}], Step [{}/{}], Predict: {} Loss: {:.4f}"
        #            .format(epoch+1, EPOCHS, i+1, total_step, prec, loss.item()))

    print("Epoch: {}, predict: {}\%, Loss: {:.4f}".format(epoch+1, prec, loss.item()))

    if (epoch + 1) % 50 == 0:
        curr_lr /= 10
        update_lr(optimizer, curr_lr)

    ResNet.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = ResNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch{}_Test:Accuracy of the model on the test images: {} %\n'.format(epoch+1, 100 * correct / total))

    ResNet.train()


predic = 100 * correct / total
torch.save(ResNet.state_dict(), 'ResNet_'+predic+'.ckpt')