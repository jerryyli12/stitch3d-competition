import os
import h5py
import numpy as np

import torch


class ScanObjectNN(torch.utils.data.Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.root = config.data_root
        self.subset = subset
        
        fileset = ['25_norot', '25rot', 'rot', 'rot_scale75']
        if self.subset == 'train':
            points = []
            labels = []
            dataset_mult = []
            for i, f in enumerate(fileset):
                h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmented' + f + '.h5'), 'r')
                points.append(np.array(h5['data']).astype(np.float32))
                labels.append(np.array(h5['label']).astype(int))
                dataset_mult.append((i + 1) * np.ones(labels[-1].shape))
                h5.close()
            self.points = np.concatenate(points)
            self.labels = np.concatenate(labels)
            self.dataset_mult = np.concatenate(dataset_mult)
        elif self.subset == 'test':
            points = []
            labels = []
            dataset_mult = []
            for i, f in enumerate(fileset):
                h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmented' + f + '.h5'), 'r')
                points.append(np.array(h5['data']).astype(np.float32))
                labels.append(np.array(h5['label']).astype(int))
                dataset_mult.append((i + 1) * np.ones(labels[-1].shape))
                h5.close()
            self.points = np.concatenate(points)
            self.labels = np.concatenate(labels)
            self.dataset_mult = np.concatenate(dataset_mult)

            #//idx = np.random.choice(self.points.shape[0], 300, replace=False)
            #self.points = self.points[idx]
            #self.labels = self.labels[idx]
            #self.dataset_mult = self.dataset_mult[idx]
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')
        print(self.labels.shape, self.dataset_mult.shape)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        current_points = current_points[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
    
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        dataset_mult = self.dataset_mult[idx]
        
        return current_points, label, dataset_mult

    def __len__(self):
        return self.points.shape[0]
