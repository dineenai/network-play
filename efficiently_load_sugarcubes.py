import os
import nibabel as nib
import numpy as np


# load sugarcube mask (nii file)
files_dir = "/home/ainedineen/motion_dnns/sugarcubes/sarah_sim"
# nii_mask = os.path.join(files_dir, 'sugar_cube_mask_resliced.nii.gz')


# create method that takes binary_mask_nii_file and size of cubes
# returns binary cubes <class 'numpy.ndarray'> 1-dim

nii_mask = os.path.join(files_dir, 'brainMask_resliced12dofmask.nii.gz')
size = 6

def get_usable_cubes_from_binary_mask(size, nii_mask):
    
    img = nib.load(nii_mask)
    # print(img.shape) #(66, 66, 42)
    # print(img.get_data_dtype()) #float64
    # print(img.affine.shape) #(4, 4)

    # access to the image data as a NumPy array
    data = img.get_fdata()
    # print(data.shape) #(66, 66, 42)
    # print(data.dtype) #float64
    # print(np.unique(data)) #[0. 1.] #binary mask

    # ideally would not be adding 6 where 0 but leave for now...
    x_pad = size-data.shape[0]%6
    y_pad = size-data.shape[1]%6
    z_pad = size-data.shape[2]%6

    npad_mask = ((0, x_pad), (0, y_pad), (0, z_pad))
    # print(npad_mask) #((0, 2), (0, 2), (0, 6))

    padded = np.pad(data, pad_width=npad_mask, mode='constant', constant_values=0)

    padded_x, padded_y, padded_z = padded.shape
    # should all be ints as padded to be multiple of 6!
    # print(padded.shape)

    new_ax_0 = padded_x//size
    new_ax_2 = padded_y//size
    new_ax_4 = padded_z//size

    new_shape = [new_ax_0, size, new_ax_2, size, new_ax_4, size]
    # padded_mask_reshaped_into_cube_dims = np.reshape(padded, [padded_x//6, 6, padded_y//6, 6, padded_z//6, 6])
    padded_mask_reshaped_into_cube_dims = np.reshape(padded, new_shape) #[11, 6, 11, 6, 7, 6]
    # print(padded_mask_reshaped_into_cube_dims.shape)

    # binary mask so if sum across all dimentions of a sugar cube (the three axes of length 6, should get 216 in locations hwere there is only brain mask - these are the sugar cubes)
    cube_totals=padded_mask_reshaped_into_cube_dims.sum(axis=5).sum(axis=3).sum(axis=1)
    # print(cube_totals.shape) #(11, 11, 7)

    # returns True where values is 216 ie 
    usable_sugarcubes=cube_totals==size*size*size

    # print(usable_sugarcubes.shape) #(11, 11, 7)

    # # verify the presence of True values
    # print(np.unique(usable_sugarcubes)) #[False  True]  
    # print(np.unique(usable_sugarcubes, return_counts=True)) #(array([False,  True]), array([583, 264]))
    counts = np.unique(usable_sugarcubes, return_counts=True)
    # print(type(counts))
    # print(counts[1][1])
    print(f'No. of usable sugarcubes: {counts[1][1]} / {counts[1][0] + counts[1][1]}')


    usable_sugarcubes=np.ravel(usable_sugarcubes)
    # print(usable_sugarcubes)
    # print(usable_sugarcubes.shape) #(847,)


    # test chopping up dataset: (CAUTION this code is not part of this function!) ####
    # chopped = usable_sugarcubes[600:,] #(array([False,  True]), array([228,  19]))
    # chopped = usable_sugarcubes[500:,] #(array([False,  True]), array([269,  78]))
    # print(np.unique(chopped, return_counts=True)) #(array([False,  True]), array([583, 264]))
    # print(chopped.shape) #(347,)
    # print(type(usable_sugarcubes)) #<class 'numpy.ndarray'>

    return usable_sugarcubes


# Load brain and choose sugar cubes based on above.....


# ultimately this function should call the function above so that there is no need to call the above function too!
def possible_sugarcubes_from_brain(size, usable_sugarcubes, file, files_dir):

    nii_brain = os.path.join(files_dir, file)
    
    brain_img = nib.load(nii_brain)
    brain_data = brain_img.get_fdata()
    
    # print(f'SHAPE pre:{brain_data.shape}')
    # print(len(brain_data.shape))

    x_pad = size-brain_data.shape[0]%6
    y_pad = size-brain_data.shape[1]%6
    z_pad = size-brain_data.shape[2]%6

    npad_brain = ((0, x_pad), (0, y_pad), (0, z_pad), (0,0)) #brain
    # npad_brain = ((0, x_pad), (0, y_pad), (0, z_pad)) #activation
    # print(npad_brain)
    padded_brain_data = np.pad(brain_data, pad_width=npad_brain, mode='constant', constant_values=0)
    # print(padded_brain_data.shape)

    padded_x, padded_y, padded_z, scan_no = padded_brain_data.shape

    # padded_mask_reshaped_into_cube_dims
    # why does this have an extra dimention: scan number!

    new_ax_0 = padded_x//size
    new_ax_2 = padded_y//size
    new_ax_4 = padded_z//size

    brain_shape_cube_dims = [new_ax_0, size, new_ax_2, size, new_ax_4, size, scan_no]
    # print(brain_shape_cube_dims)

    brain_reshaped_into_cube_dims = np.reshape(padded_brain_data, brain_shape_cube_dims)
    # print(brain_reshaped_into_cube_dims.shape)

    # THINK ABOUT THIS BIT
    # number of cube width, height, depth, cube sides, cube sides, cube sides, volumes (first 3 are kind of indexes, second three are actual values at each index)
    brain_reshaped_into_cube_dims_transpose=np.transpose(brain_reshaped_into_cube_dims,[0,2,4,1,3,5,6])
    brain_reshaped_to_index_all_cubes=np.reshape(brain_reshaped_into_cube_dims_transpose, [new_ax_0*new_ax_2*new_ax_4, size,size,size,-1])
    # print(brain_reshaped_to_index_all_cubes.shape) #(847, 6, 6, 6, 24)

    # use usable_sugarcubes to pick out only cubes that have all 1s, will have True value!
    brain_reshaped_to_index_brain_cubes=brain_reshaped_to_index_all_cubes[usable_sugarcubes,:,:,:,:]
    # print(brain_reshaped_to_index_brain_cubes.shape) #(847, 6, 6, 6, 24)
    # print(brain_reshaped_into_cube_dims_transpose)

    return brain_reshaped_to_index_brain_cubes




# add parameter to give the relavant activation magnitude!!!
def get_act_pattern_of_cubes(size, usable_sugarcubes, file, files_dir):

    nii_act= os.path.join(files_dir, file)
    
    act_img = nib.load(nii_act)
    act_data = act_img.get_fdata()
    
    # print(f'SHAPE pre:{brain_data.shape}')
    # print(len(brain_data.shape))

    x_pad = size-act_data.shape[0]%6
    y_pad = size-act_data.shape[1]%6
    z_pad = size-act_data.shape[2]%6

    # npad_brain = ((0, x_pad), (0, y_pad), (0, z_pad), (0,0)) #brain
    npad_act = ((0, x_pad), (0, y_pad), (0, z_pad)) #activation
    # print(npad_act)
    padded_act_data = np.pad(act_data, pad_width=npad_act, mode='constant', constant_values=0)
    # print(padded_act_data.shape)

    padded_x, padded_y, padded_z = padded_act_data.shape

    # padded_mask_reshaped_into_cube_dims
    # why does this have an extra dimention: scan number!

    new_ax_0 = padded_x//size
    new_ax_2 = padded_y//size
    new_ax_4 = padded_z//size

    brain_shape_cube_dims = [new_ax_0, size, new_ax_2, size, new_ax_4, size]
    # print(brain_shape_cube_dims)

    act_reshaped_into_cube_dims = np.reshape(padded_act_data, brain_shape_cube_dims)
    # print(act_reshaped_into_cube_dims.shape)

    # THINK ABOUT THIS BIT
    # number of cube width, height, depth, cube sides, cube sides, cube sides, volumes (first 3 are kind of indexes, second three are actual values at each index)
    act_reshaped_into_cube_dims_transpose=np.transpose(act_reshaped_into_cube_dims,[0,2,4,1,3,5])
    act_reshaped_to_index_all_cubes=np.reshape(act_reshaped_into_cube_dims_transpose, [new_ax_0*new_ax_2*new_ax_4, size,size,size])
    # print(act_reshaped_to_index_all_cubes.shape) #(847, 6, 6, 6, 24)

    # use usable_sugarcubes to pick out only cubes that have all 1s, will have True value!
    act_reshaped_to_index_brain_cubes=act_reshaped_to_index_all_cubes[usable_sugarcubes,:,:,:]
    print(f'Activation pattern of usable sugarcubes: {act_reshaped_to_index_brain_cubes.shape}')#(847, 6, 6, 6, 24)

    return act_reshaped_to_index_brain_cubes



    # to select a cube - choose brain volume
    # and seleced cube between 0 and 263
    # chosen_cube = 0
    # chosen_vol = 0

def choose_a_cube(usable_sugarcubes, brain_reshaped_to_index_brain_cubes,  activation_pattern_of_possible_sugarcubes, chosen_cube, chosen_vol, save_cube_name=''):
    
    chosen_sugarcube_of_brain = brain_reshaped_to_index_brain_cubes[chosen_cube,:,:,:,chosen_vol]
    # corresponding activation
    chosen_sugarcube_of_brain_act_pattern = activation_pattern_of_possible_sugarcubes[chosen_cube,:,:,:]
    

    save_dir = os.path.join(files_dir, 'sample_cubes')
    
    save_file = save_cube_name + f'_cube{chosen_cube}_vol{chosen_vol}.nii'
    # corresponding activation
    save_act_pattern =save_cube_name + f'_cube{chosen_cube}_binary_activation.nii'
    
    save_path = os.path.join(save_dir, save_file)
    # corresponding activation
    save_act_path = os.path.join(save_dir, save_act_pattern)

    img_to_save = nib.Nifti1Image(chosen_sugarcube_of_brain, affine=np.eye(4))
    nib.save(img_to_save, save_path)
    
    # corresponding activation
    act_to_save = nib.Nifti1Image(chosen_sugarcube_of_brain_act_pattern, affine=np.eye(4))
    nib.save(act_to_save, save_act_path)

    return chosen_sugarcube_of_brain



usable_sugarcubes = get_usable_cubes_from_binary_mask(size, nii_mask)


activation_pattern_of_possible_sugarcubes = get_act_pattern_of_cubes(size, usable_sugarcubes, 'striping_resliced12dofmask.nii.gz', files_dir)
print(f'Activation pattern of usable sugarcubes: {activation_pattern_of_possible_sugarcubes.shape}')


no_motion_no_noise_possible_sugarcubes = possible_sugarcubes_from_brain(size, usable_sugarcubes, 'simStripesThick_noMotion_sinAct_1pct_noNoise.nii.gz', files_dir)
print(f'Array indexing possible sugarcubes: {no_motion_no_noise_possible_sugarcubes.shape}')
choose_a_cube(usable_sugarcubes, no_motion_no_noise_possible_sugarcubes, activation_pattern_of_possible_sugarcubes, chosen_cube=74, chosen_vol=3, save_cube_name='no_motion_no_noise')



# # no_motion_noise = possible_sugarcubes_from_brain(size, usable_sugarcubes, 'simStripesThick_noMotion_sinAct_1pct_SNR50.nii.gz', 74, 2, files_dir)
# # print(no_motion_noise.shape)
# # choose_a_cube(1, 1, usable_sugarcubes, no_motion_noise, activation_pattern_of_possible_sugarcubes, 'no_motion_noise' )






