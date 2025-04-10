from torch.utils.data import Dataset
import torch
import numpy as np
import numba as nb


def get_map(x):
    class_map = {40: 2}
    return class_map.get(x, 0)

class KittiDataset(Dataset):
    def __init__(self, filenames,
                 scene_size=5000):
        self.scene_size = scene_size
        self.map_classes = np.vectorize(
            get_map,
        )
        self.filenames = filenames

    
    def read_labels(self, filename, filename_manual, drop_road=False):
        labels_road = (np.fromfile(filename, dtype=np.int32) & 0xFFFF).reshape((-1, 1)).astype(np.uint8)
        labels_road = self.map_classes(labels_road)[:, 0]
        labels_plane = np.load(filename_manual)

        result_labels = np.clip(labels_road + labels_plane, 0, int(drop_road) + 1).astype(np.uint8)

        return torch.from_numpy(result_labels[None].T)
    
    def read_scene(self, filename):
        np_array = np.fromfile(
            filename, 
            dtype=np.float32
        ).reshape(-1, 4)

        return torch.from_numpy(np_array)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        scene_path, labels_path, manual_path = self.filenames[idx]
        scene = self.read_scene(scene_path)
        labels = self.read_labels(labels_path, manual_path)
        indices = np.random.choice(range(len(labels)),
                                size=self.scene_size)
        return scene[indices], labels[indices]


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


class spherical_dataset(Dataset):
  def __init__(self, in_dataset, grid_size, 
               ignore_label = 255, fixed_volume_space = False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        xyz, labels = data
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers #TODO: calculate relative coordinate using polar system?
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        return_fea = return_xyz

        data_tuple += (grid_ind, labels, return_fea) 

        return data_tuple

def collate_fn_BEV(data): # stack alone batch dimension
    data2stack=np.stack([d[0] for d in data]).astype(np.float32) # grid-wise coor
    label2stack=np.stack([d[1] for d in data])                   # grid-wise sem label
    grid_ind_stack = [d[2] for d in data]                        # point-wise grid index
    point_label = [d[3] for d in data]                           # point-wise sem label
    xyz = [d[4] for d in data]                                   # point-wise coor

    return {
        'vox_coor': torch.from_numpy(data2stack),
        'vox_label': torch.from_numpy(label2stack),
        'grid': grid_ind_stack,
        'pt_labs': point_label,
        'pt_fea': xyz,
    }

def build_dataloader(filenames, scene_size, grid_size=[480, 360, 32], batch_size=2):
    train_pt_dataset = KittiDataset(filenames, scene_size=scene_size)
    s_dataset = spherical_dataset(train_pt_dataset, grid_size)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset = s_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn_BEV,
        shuffle=True,
    )

    return train_dataset_loader
