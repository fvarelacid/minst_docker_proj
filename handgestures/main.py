import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision.transforms.transforms import Grayscale
from handgestures.transform_image import transform_single_image
import numpy as np
from PIL import Image

# def get_data():
#     data_dir = '/Users/franciscovarelacid/Desktop/Strive/images_ml/'
    
#     transform = transforms.Compose([
#     transforms.Resize(128),
#     transforms.Grayscale(),
#     transforms.ToTensor()])

#     train_set = datasets.ImageFolder(data_dir + '/training_set', transform=transform)
#     test_set = datasets.ImageFolder(data_dir + '/test_set', transform=transform)

#     train = DataLoader(train_set, batch_size=32, shuffle=True)
#     test = DataLoader(test_set, batch_size=32, shuffle=True)

#     return train, test

# def train_imshow():
#     classes = ('paper', 'rock', 'scissors') # Defining the classes we have
#     dataiter = iter(train)
#     images, labels = dataiter.next()
#     fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
#     for i in range(5):
#         ax = axes[i]
#         ax.imshow(images[i].permute(1, 2, 0)) 
#         ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
#     plt.show()

# train, test = get_data()

# train_imshow()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# if torch.cuda.is_available(): # Checking if we can use GPU
#     model = net.cuda()
#     criterion = criterion.cuda()


# def train_net(n_epoch): # Training our network
#     losses = []
#     for epoch in range(n_epoch):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in enumerate(train, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             losses.append(loss)
#             running_loss += loss.item()
#             if i % 100 == 0:  # print every 100 mini-batches
#                 print('[%d, %5d] loss: %.10f' %
#                       (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0
#     losses_list = [fl.item() for fl in losses]
#     plt.plot(losses_list, label='Training loss')
#     plt.show()
#     print('Finished Training')

# # train_net(10)

# PATH = './RPS_net.pth'
# torch.save(net.state_dict(), PATH)

# # Loading the trained network
# net.load_state_dict(torch.load(PATH))
# net.eval()

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the %d test images: %d %%' % (len(test),
#     100 * correct / total))

# transform = transforms.Compose([
#     transforms.Resize(128),
#     transforms.ToTensor()])


# Disable grad
# with torch.no_grad():
#     image = transform_single_image('handgestures/image.jpg')
#     pil_image = Image.fromarray(image)
#     img = transform(pil_image)

#     # Generate prediction
#     prediction = net(img)
    
#     # Predicted class value using argmax
#     predicted_class = np.argmax(prediction)
    
#     # Reshape image
#     image = image.reshape(28, 28, 1)
    
#     # Show result
#     plt.imshow(image, cmap='gray')
#     plt.title(f'Prediction: {predicted_class}')
#     plt.show()