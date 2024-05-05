import torch
import torchvision

import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_Res=True):
        super(ResBlock, self).__init__()
        self.use_Res = use_Res
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or not use_Res:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.use_Res:
            z = self.shortcut(x)
            y = y + z

        y = self.relu(y)
        return y


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = ResNet18(num_classes=100).to(device)

def train( config ):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                wandb.log({"epoch": epoch, "loss": loss.item()})

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

if 'trainset' not in globals():
    #數據歸ㄧ化＋轉換ndarray，標準化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #download dataset
    full_dataset = torchvision.datasets.CIFAR100( root='./data', train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR100( root='./data', train=False, transform=transform, download = True)
    classes = full_dataset.classes
    trainset, validset = torch.utils.data.random_split( full_dataset, [40000, 10000])

    #speedup, pid （數據封裝）
    trainloader = torch.utils.data.DataLoader( trainset, batch_size = 16, shuffle = True )
    validloader = torch.utils.data.DataLoader( validset, batch_size = 16, shuffle = True )
    testloader = torch.utils.data.DataLoader( testset, batch_size = 16, shuffle = False )


model = ResNet18(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()


def main() :
    # 使用weights & biases 監控

    wandb.init(
        project="mvc_hw1",
        mode= "online",
        config={
            "learning_rate": 0.1,
            "architecture": "cnn",
            "dataset": "CIFAR-100",
            "batch_size": 16,
            "epochs": 20
        }
    )

    train( wandb.config )
    test()
    wandb.finish()

if __name__ == '__main__' :
    main()
