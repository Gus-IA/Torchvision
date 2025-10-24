import torch
import torchvision
import random
import matplotlib.pyplot as plt

# descargamos los datos de train y test
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# pintamos una muestra
r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
for _r in range(r):
    for _c in range(c):
        ax = plt.subplot(r, c, _r*c + _c + 1)
        ix = random.randint(0, len(trainset))
        img, label = trainset[ix]
        plt.axis("off")
        plt.imshow(img)
plt.tight_layout()
plt.show()


# transformaciones, transformarla a un tensor y normalizarla
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

# nuevas transformaciones aleatorias
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((28,28)),
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.RandomHorizontalFlip(),
        # ...
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)


r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
for _r in range(r):
    for _c in range(c):
        ax = plt.subplot(r, c, _r*c + _c + 1)
        ix = 10
        img, label = trainset[ix]
        plt.axis("off")
        # desnormalizar
        img = img*0.5 + 0.5
        img = img.permute(1, 2, 0)
        plt.imshow(img)
plt.tight_layout()
plt.show()


# descargamos el modelo resnet18
resnet = torchvision.models.resnet18()
print(resnet)


# red preentrenada
resnet = torchvision.models.resnet18(pretrained=True)

# cambiamos el número de la última capa
num_classes = 10
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
print(resnet)