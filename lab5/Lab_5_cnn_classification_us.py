import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\anton\Documents\data\data\train', transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\anton\Documents\data\data\test', transform=data_transforms)

class_names = train_dataset.classes
print("Обнаруженные классы:", train_dataset.classes)
batch_size = 6
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = torchvision.models.alexnet(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

num_classes = len(class_names)
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc', nn.Linear(4096, num_classes))
net.classifier = new_classifier
net = net.to(device)

correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy = 100 * correct_predictions / num_test_samples
print(f'Точность модели до обучения: {accuracy}%')

num_epochs = 10
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
save_loss = []

t = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = lossFn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_loss.append(loss.item())
        if i % 10 == 0:
            print(f'Эпоха {epoch} из {num_epochs}, Шаг {i}, Ошибка: {loss.item()}')

print(f'Время обучения: {time.time() - t} секунд')

plt.figure()
plt.plot(save_loss)
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.title('График обучения')
plt.show()

correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy = 100 * correct_predictions / num_test_samples
print(f'Точность модели после обучения: {accuracy}%')

inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i, j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)