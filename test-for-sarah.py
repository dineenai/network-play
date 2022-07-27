import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from pathlib import Path

batch_size = 100
inputsize = 20
train_nexamples = 10000 # number of training stimuli
test_nexamples = 1000 # number of test stimuli

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, modeltype):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        if modeltype=='onelinear':
            self.layer_stack = nn.Sequential(
                nn.Linear(inputsize, 1),
            )
        elif modeltype=='conv1d':
            self.layer_stack = nn.Sequential(
                nn.Conv1d(1, 4, 3, padding=2),
                nn.ReLU(),
#                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Linear(22, 1),
              
            )
        elif modeltype=='twolinear':
            self.layer_stack = nn.Sequential(
                nn.Linear(inputsize, inputsize),
                nn.ReLU(),
                nn.Linear(inputsize, 1),
              
            )

    def forward(self, x):
        outputs = self.layer_stack(x)
        return outputs

def get_dataset(nexamples, batch_size, figname=None, noisetype=None):
    my_x = np.random.randn(nexamples, 1, inputsize) 
    my_y = np.random.randn(nexamples, 1, 1 )
    my_noise = (1+np.random.randn(nexamples, 1, 1))*4

    my_x[:, :, 10:] += my_y

    if noisetype=='evenvoxels':
        my_x[:, :, ::2] += my_noise

    if noisetype=='oddevenvoxelsrandom':
        my_x[:nexamples//2, :, ::2] += my_noise[:nexamples//2,:]
        my_x[nexamples//2:, :, 1::2] += my_noise[nexamples//2:,:]

    if figname:
        fig,ax = plt.subplots(ncols=3, sharey=True)
        ax[0].imshow(my_x[:,0,:], aspect='auto', interpolation='none')
        ax[0].set_xlabel('voxels')
        ax[0].set_ylabel('timepoints')
        ax[1].imshow(my_y, aspect='auto', interpolation='none')
        ax[1].set_title('activation')
        ax[2].imshow(my_noise, aspect='auto', interpolation='none')
        ax[2].set_title('noise')
        plt.savefig(figname)

    tensor_x = torch.Tensor(my_x) # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset

    return DataLoader(my_dataset, batch_size=batch_size, shuffle=True) # create your dataloader



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} ")
    return test_loss

def test_ideal(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')
            pred = torch.mean(X[:,:,10:], axis=2, keepdim=True)
            test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    return test_loss

if __name__=='__main__':

    Path("graphs").mkdir(parents=True, exist_ok=True)
    for networktype in ['onelinear','twolinear']:
        for noisetype in ['none', 'evenvoxels','oddevenvoxelsrandom']:
            model = NeuralNetwork(networktype).to(device)
            print(model)

            train_dataloader = get_dataset(train_nexamples, batch_size, f'graphs/network-{networktype}_noise-{noisetype}_train_data.png', noisetype)
            test_dataloader = get_dataset(test_nexamples, batch_size, None, noisetype)

            loss_fn = nn.L1Loss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            epochs = 200
            test_loss=[]
            for t in range(epochs):
                print(f"Epoch {t+1} ", end='')
                train(train_dataloader, model, loss_fn, optimizer)
                test_loss.append(test(test_dataloader, model, loss_fn))

            # ideal model of averaging 10 onwards
            ideal_loss=test_ideal(test_dataloader, model, loss_fn)

            fig, ax =plt.subplots(nrows=len(list(model.named_parameters()))//2+1)
            figind = 0
            for ind, (name, param) in enumerate(model.named_parameters()):
                if 'weight' in name:
                    pars = param.cpu().detach().numpy()
                    ax[figind].plot(pars.ravel())
                    ax[figind].set_title(name)
                    figind+=1


            plt.savefig(f'graphs/network-{networktype}_noise-{noisetype}_model_parameters.png')

            plt.figure()
            plt.plot(test_loss)
            plt.xlabel('Epoch')
            plt.ylabel('L1 test loss')

            plt.plot([0, epochs],[ideal_loss, ideal_loss],'g--')
            plt.savefig(f'graphs/network-{networktype}_noise-{noisetype}_loss.png')

            print(f'Ideal model, loss= {ideal_loss}')

            print("Done!")