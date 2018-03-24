import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import json
import os.path

# Configuration
output_dir = "out"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 50
learning_rate = 1e-2

# Output data
out_data = {}
out_data['type'] = 'optimized'
out_data['num_epochs'] = num_epochs
out_data['batch_size'] = batch_size
out_data['learning_rate'] = learning_rate

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model
out_data['loss'] = {'epoch':[], 'batch':[], 'loss':[]}
average_epoch_loss = []
for epoch in range(num_epochs):
    num_batches = 0
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        class_probs = net(images)
        loss = criterion(class_probs, labels)
        loss.backward()
        optimizer.step()
        print("epoch:{} batch:{} --> loss:{}".format(epoch, i, loss.data.numpy()))
        out_data['loss']['epoch'].append(epoch)
        out_data['loss']['batch'].append(i)
        out_data['loss']['loss'].append(loss.data.numpy().item())
        num_batches += 1
        total_loss += loss.data.numpy()
    average_epoch_loss.append(total_loss/num_batches)
        

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    
    class_probs = net(images)
    pred = class_probs.data.max(1, keepdim=True)[1]
    correct += pred.eq(labels.view_as(pred)).long().sum()
    
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
out_data['accuracy'] = (100 * correct / total)

# Save the Model
model_dir = os.path.join(output_dir, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(net.state_dict(), os.path.join(model_dir, '{}_model.pkl'.format('optimized')))

with open("out/mnist_optimized_output_data.json", 'wt') as f:
    json.dump(out_data, f)
