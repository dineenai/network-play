import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from pathlib import Path

import nibabel as nib
import os
import pandas as pd

from efficiently_load_sugarcubes import get_usable_cubes_from_binary_mask
from efficiently_load_sugarcubes import possible_sugarcubes_from_brain
from efficiently_load_sugarcubes import get_act_pattern_of_cubes

files_dir = "/home/ainedineen/motion_dnns/sugarcubes/sarah_sim"
nii_mask = os.path.join(files_dir, 'brainMask_resliced12dofmask.nii.gz')
act_info = os.path.join(files_dir, 'activation')

size = 6

usable_sugarcubes = get_usable_cubes_from_binary_mask(6, nii_mask)
print(f'usable_sugarcubes: {usable_sugarcubes.shape}')



no_motion_no_noise = possible_sugarcubes_from_brain(size, usable_sugarcubes, 'simStripesThick_noMotion_sinAct_1pct_noNoise.nii.gz', files_dir)
print(no_motion_no_noise.shape)

# my_y is activation

est_act = os.path.join(act_info, 'est_act_per_vol.csv')

# test load
est_activation = pd.read_csv(est_act)



activation_pattern = get_act_pattern_of_cubes(size, usable_sugarcubes, 'striping_resliced12dofmask.nii.gz', files_dir)
print(activation_pattern.shape)


batch_size = 100

inputsize = size*size*size #(no of voxels)


train_test_split = 264 #roughly 30% test set
# train_nexamples = 10000 # number of training stimuli
# test_nexamples = 1000 # number of test stimuli
# 

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

# def get_dataset(ground_truth, est_activation, train_test_split, batch_size, mode=['test', 'train'], figname=None):
def get_dataset(ground_truth, est_activation, train_test_split, batch_size, chosen_vol, figname=None):
    print(ground_truth.shape)

    # print(f'Mode is {mode}')
    # this is for train
    
    # if mode =='train':

    print('here')
    chosen_vol = 7 #Manual for now
    # train
    # my_x = no_motion_no_noise[:train_test_split,  :, :, :, chosen_vol]
    # test
    # my_x = no_motion_no_noise[train_test_split:, :, :, :, chosen_vol]
    
    my_x = no_motion_no_noise[:,  :, :, :, chosen_vol]
    print(my_x.size)
    # flat_my_x = my_x.reshape(264, -1)
    flat_my_x = my_x.reshape(264, 1, (size*size*size)) #(264, 1, 216)
    print(f'Shape of my_x: {my_x.shape}')
    print(f'flat_my_x {flat_my_x.shape}') #flat_my_x (264, 216)

    
    print(f'Estimated activation for vol {chosen_vol} is {est_activation.est_activation[chosen_vol]}')
    
    act_for_vol = est_activation.est_activation[chosen_vol]
    print(act_for_vol)

    # activation value for chosen volume
    flat_my_y = np.full((264, 1, 1), act_for_vol) # (264, 1, 1)

    print(f'flat_my_y: {flat_my_y.shape}')

    
    tensor_x = torch.Tensor(flat_my_x) # transform to torch tensor
    tensor_y = torch.Tensor(flat_my_y) #torch.Size([264, 1])
    print(f"Tensors(shape): x{tensor_x.shape}, y{tensor_y.shape}")


    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    print(f'len my_dataset: {len(my_dataset)}')

    return DataLoader(my_dataset, batch_size=batch_size, shuffle=True) # create your dataloader

    



    # if figname:
    #     fig,ax = plt.subplots(ncols=3, sharey=True)
    #     ax[0].imshow(my_x[:,0,:], aspect='auto', interpolation='none')
    #     ax[0].set_xlabel('voxels')
    #     ax[0].set_ylabel('timepoints')
    #     ax[1].imshow(my_y, aspect='auto', interpolation='none')
    #     ax[1].set_title('activation')
    #     ax[2].imshow(my_noise, aspect='auto', interpolation='none')
    #     ax[2].set_title('noise')
    #     plt.savefig(figname)





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
        model = NeuralNetwork(networktype).to(device)
        print(model)

        train_dataloader = get_dataset(no_motion_no_noise, est_activation, train_test_split, batch_size, 7, f'graphs/network-{networktype}_train_data.png')
        test_dataloader = get_dataset(no_motion_no_noise, est_activation, train_test_split, batch_size, 14, f'graphs/network-{networktype}_test_data.png')
        # test_dataloader = get_dataset(no_motion_no_noise, est_activation, train_test_split, batch_size, 'test')


        print(train_dataloader)

        # # TRY ADDING
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


        plt.savefig(f'graphs/network-{networktype}_model_parameters.png')

        plt.figure()
        plt.plot(test_loss)
        plt.xlabel('Epoch')
        plt.ylabel('L1 test loss')

        plt.plot([0, epochs],[ideal_loss, ideal_loss],'g--')
        plt.savefig(f'graphs/network-{networktype}_loss.png')

        print(f'Ideal model, loss= {ideal_loss}')

        print("Done!")