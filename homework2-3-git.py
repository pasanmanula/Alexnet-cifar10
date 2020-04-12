
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Network(nn.Module):
    
    def __init__(self):
        super().__init__() 
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=85, kernel_size=5),
            nn.BatchNorm2d(85),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(85*5*5, 2125),
            nn.BatchNorm1d(2125),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2125, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(500, 10)
        )

    def forward(self, x):

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# net = Net()
network = Network().to(device)

# ==============================================DataSet Start==================
# Training set
train_set = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = 100,
    shuffle=False,
    num_workers=2
)

total_sample_train_data = len(train_set) #for Cifar10

# Test set

test_set = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)


test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = 4,
    shuffle=False,
    num_workers=2
)

total_sample_test_data = len(test_set) #for Cifar10


classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# ==============================================DataSet END==================



optimizer = optim.RMSprop(network.parameters(), lr=0.001,alpha=0.99, eps=1e-08, weight_decay=0.1, momentum=0, centered=False)

# optimizer = optim.Adam(network.parameters(), lr=0.001)

for epoch in range(20):
    print("Epoch started :",epoch)
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images,labels = batch

        images, labels = images.to(device), labels.to(device)

        preds = network(images) #Forward_Pass

        loss = F.cross_entropy(preds, labels) #Calc loss
        optimizer.zero_grad() #Zero-out any remaining gradients in placeholders
        loss.backward() #Calc gradients
        optimizer.step() # Updating the weights

        total_loss += loss.item() #Update batch loss
        total_correct += get_num_correct(preds, labels)

    total_loss = total_loss / len(train_loader)
    print("Epoch: ",epoch," Total Correct: ",total_correct,"Loss: ",total_loss)

print("Training Finished!")

# # Saving the learned weights
PATH = './cifar10_uofm_rmsprop.pth'
torch.save(network.state_dict(), PATH)


# Testing the network
dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

network = Network()
network.load_state_dict(torch.load(PATH))

outputs = network(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# Building the confusion matrix
@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(test_set, batch_size=10000)
    test_preds = get_all_preds(network, prediction_loader)

test_labelsd = torch.LongTensor(test_set.test_labels)
stacked = torch.stack(
    (
        test_labelsd
        ,test_preds.argmax(dim=1)
    )
    ,dim=1
)
confusion = torch.zeros(10,10, dtype=torch.int32)

for p in stacked:
    tl, pl = p.tolist()
    confusion[tl, pl] = confusion[tl, pl] + 1

print(confusion)

# # Visualizing kernals.
# weight = network.conv_layer[0].weight.data.numpy()
# print(np.shape(weight[0]))


# plt.figure()

# #subplot(r,c) provide the no. of rows and columns
# f, axarr = plt.subplots(4,5) 

# # use the created array to output your multiple images. In this case I have stacked 4 images vertically
# axarr[0,0].imshow(weight[0,0,:,:],cmap='gray')
# axarr[0,1].imshow(weight[0,1,:,:],cmap='gray')
# axarr[0,2].imshow(weight[2,0,:,:],cmap='gray')
# axarr[0,3].imshow(weight[3,0,:,:],cmap='gray')
# axarr[0,4].imshow(weight[4,0,:,:],cmap='gray')

# axarr[1,0].imshow(weight[5,0,:,:],cmap='gray')
# axarr[1,1].imshow(weight[6,1,:,:],cmap='gray')
# axarr[1,2].imshow(weight[7,0,:,:],cmap='gray')
# axarr[1,3].imshow(weight[8,0,:,:],cmap='gray')
# axarr[1,4].imshow(weight[9,0,:,:],cmap='gray')

# axarr[2,0].imshow(weight[10,0,:,:],cmap='gray')
# axarr[2,1].imshow(weight[11,1,:,:],cmap='gray')
# axarr[2,2].imshow(weight[12,0,:,:],cmap='gray')
# axarr[2,3].imshow(weight[13,0,:,:],cmap='gray')
# axarr[2,4].imshow(weight[14,0,:,:],cmap='gray')

# axarr[3,0].imshow(weight[15,0,:,:],cmap='gray')
# axarr[3,1].imshow(weight[16,1,:,:],cmap='gray')
# axarr[3,2].imshow(weight[17,0,:,:],cmap='gray')
# axarr[3,3].imshow(weight[18,0,:,:],cmap='gray')
# axarr[3,4].imshow(weight[19,0,:,:],cmap='gray')


# plt.show()
# plt.hold(True)