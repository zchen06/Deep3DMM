import argparse, os, pdb
import os.path as osp
import glob
import pickle
import copy
import numpy as np
import h5py
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
from tqdm import tqdm
from math import ceil

from psbody.mesh import Mesh
from utils import get_vert_connectivity
from transform import Normalize

class ComaDataset(Dataset):
    def __init__(self, root_dir, dataset_dir=None, dtype='train', split='sliced', split_term='sliced', nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_tranform = pre_transform

        dest_dir = osp.join(self.root_dir, 'processed', self.split_term)
        train_fname = osp.join(dest_dir, 'train_file_list.npy')
        test_fname = osp.join(dest_dir, 'test_file_list.npy')
        val_fname = osp.join(dest_dir, 'val_file_list.npy')
        if os.path.exists(train_fname) is False:
            self.data_file = self.get_datafile(self.root_dir)
            self.datafile_train, self.datafile_val, self.datafile_test = self.get_datafile_subset()

            if os.path.exists(dest_dir) is False:
                os.makedirs(dest_dir)
            np.save(train_fname, self.datafile_train)            
            np.save(test_fname, self.datafile_test)            
            np.save(val_fname, self.datafile_val)
        else:
            self.datafile_val = np.load(val_fname)
            self.datafile_train = np.load(train_fname)            
            self.datafile_test = np.load(test_fname)
            self.datafile_train = [str(self.datafile_train[i]) for i in range(len(self.datafile_train))]
            self.datafile_test = [str(self.datafile_test[i]) for i in range(len(self.datafile_test))]
            self.datafile_val = [str(self.datafile_val[i]) for i in range(len(self.datafile_val))]
            self.data_file = self.datafile_test + self.datafile_val + self.datafile_train            
                
        super(ComaDataset, self).__init__(root_dir, transform, pre_transform)
        self.dtype = dtype
            
        norm_path = self.processed_paths[0]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        len_train, len_val, len_test = len(self.datafile_train), len(self.datafile_val), len(self.datafile_test)
        processed_files = ['norm.pt']
        self.train_files = ['train/data_{}.pt'.format(idx) for idx in range(len_train)]
        self.test_files = ['test/data_{}.pt'.format(idx) for idx in range(len_test)]
        self.val_files = ['val/data_{}.pt'.format(idx) for idx in range(len_val)]

        processed_files += self.train_files
        processed_files += self.test_files
        processed_files += self.val_files
        processed_files = [self.split_term+'/'+pf for pf in processed_files]
        return processed_files

    def get_datafile(self, root_dir):
        # Downloaded data is present in following format root_dir/*/*/*.py
        if 'coma' in root_dir.lower():
            data_file = glob.glob(root_dir + '/*/*/*.ply')
        elif 'dfaust' in root_dir.lower():
            data_file = glob.glob(root_dir + '/objs/*/*.obj')
        else:
            raise NotImplementedError
        return data_file

    def get_datafile_subset(self):
        # print('Getting subset files....')
        datafile_train, datafile_val, datafile_test = [], [], []
        for idx, data_file in enumerate(self.data_file):        
            if self.split == 'sliced':
                if idx % 100 <= 10:
                    datafile_test.append(data_file)
                elif idx % 100 <= 20:
                    datafile_val.append(data_file)
                else:
                    datafile_train.append(data_file)
                    
            elif self.split == 'expression':
                if data_file.split('/')[-2] == self.split_term:
                    datafile_test.append(data_file)
                else:
                    datafile_train.append(data_file)

            elif self.split == 'identity':
                if data_file.split('/')[-3] == self.split_term:
                    datafile_test.append(data_file)
                else:
                    datafile_train.append(data_file)
            elif self.split == 'custom':
                test_terms = self.split_term.split('-')
                test_flag = False
                for term in test_terms:
                    if term in data_file:
                        test_flag = True
                if test_flag is True:
                    datafile_test.append(data_file)
                else:
                    datafile_train.append(data_file)
            else:
                raise Exception('sliced, expression and identity are the only supported split terms')

        if self.split != 'sliced':
            datafile_val = copy.deepcopy(datafile_test[-self.nVal:])
            datafile_test = copy.deepcopy(datafile_test[:-self.nVal])
        return datafile_train, datafile_val, datafile_test

    def save_np_vertices(self, dest_dir, vertices):
        np.save(dest_dir, np.array(vertices))
        return 0

    def process(self):
        print('Computing mean and std ...')
        train_vertices = []
        for idx, data_file in tqdm(enumerate(self.datafile_train)):
            mesh = Mesh(filename=data_file)
            train_vertices.append(mesh.v)
        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        dest_path = osp.join(self.processed_dir, self.split_term)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        torch.save(norm_dict, self.processed_paths[0])
        
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
        
        subsets = ['train', 'test', 'val']
        for subset in subsets:
            print('processing {} ...'.format(subset))
            dest_path = osp.join(self.processed_dir, self.split_term, '{}'.format(subset))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            vertices = []
            for idx, data_file in tqdm(enumerate(eval('self.datafile_'+subset))):
                mesh = Mesh(filename=data_file)
                mesh_verts = torch.Tensor(mesh.v)
                adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                edge_index = torch.LongTensor(np.vstack((adjacency.row, adjacency.col)))
                # edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
                vertices.append(mesh.v)
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                torch.save(data, osp.join(self.processed_dir, self.split_term, '{}/data_{}.pt'.format(subset,idx)))
            # self.save_np_vertices(osp.join(self.processed_dir, self.split_term, '{}'.format(subset)), vertices)
   
    def __len__(self):
        return self.len()

    def download(self):
        pass

    def len(self):
        if self.dtype == 'train':
            # return len(self.train_files) 
            return len(self.datafile_train) 
        elif self.dtype == 'test':
            return len(self.datafile_test)
        elif self.dtype == 'val':
            return len(self.datafile_val)
        else:
            raise Exception("train, val and test are supported data types")
        
    def load_data_idx(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.split_term, self.dtype, 'data_{}.pt'.format(idx)))
        return data
    
    def get(self, idx):
        data = self.load_data_idx(idx)
        return data
        

class ComaDataset_InMemory(InMemoryDataset):
    def __init__(self, root_dir, dataset_dir=None, dtype='train', split='sliced', split_term='sliced', nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_tranform = pre_transform
        # Downloaded data is present in following format root_dir/*/*/*.py
        if 'coma' in root_dir.lower():
            self.data_file = glob.glob(self.root_dir + '/*/*/*.ply')
        elif 'dfaust' in root_dir.lower():
            self.data_file = glob.glob(self.root_dir + '/objs/*/*.obj')
        else:
            raise NotImplementedError

        # for npy storage
        self.dataset_dir = os.path.join(root_dir,'processed') if dataset_dir is None else dataset_dir
        
        super(ComaDataset_InMemory, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
        processed_files = [self.split_term+'_'+pf for pf in processed_files]
        return processed_files
        
    def save_vertices(self, train_vertices, val_vertices, test_vertices):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        np.save(self.dataset_dir+'/'+self.split_term+'_train', train_vertices)
        np.save(self.dataset_dir+'/'+self.split_term+'_val', val_vertices)
        np.save(self.dataset_dir+'/'+self.split_term+'_test', test_vertices)
        
        print( "Saving ... ", self.dataset_dir)
        return 0

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_vertices, val_vertices, test_vertices = [], [], []
        for idx, data_file in tqdm(enumerate(self.data_file)):
            mesh = Mesh(filename=data_file)
            mesh_verts = torch.Tensor(mesh.v)
            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            # edge_index = torch.LongTensor(np.vstack((adjacency.row, adjacency.col)))
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

            if self.split == 'sliced':
                if idx % 100 <= 10:
                    test_data.append(data)
                    test_vertices.append(mesh.v)
                elif idx % 100 <= 20:
                    val_data.append(data)
                    val_vertices.append(mesh.v)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'expression':
                if data_file.split('/')[-2] == self.split_term:
                    test_data.append(data)
                    test_vertices.append(mesh.v)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'identity':
                if data_file.split('/')[-3] == self.split_term:
                    test_data.append(data)
                    test_vertices.append(mesh.v)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)
            elif self.split == 'custom':
                test_terms = self.split_term.split('-')
                test_flag = False
                for term in test_terms:
                    if term in data_file:
                        test_flag = True
                if test_flag is True:
                    # print('testing set: {}'.format(data_file))
                    test_data.append(data)
                    test_vertices.append(mesh.v)
                else:
                    # print('training set: {}'.format(data_file))
                    train_data.append(data)
                    train_vertices.append(mesh.v)                    
            else:
                raise Exception('sliced, expression and identity are the only supported split terms')

        if self.split != 'sliced':
            val_data = test_data[-self.nVal:]
            test_data = test_data[:-self.nVal]
            val_vertices = test_vertices[-self.nVal:]
            test_vertices = test_vertices[:-self.nVal]

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            print('Transforming data...')
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        print('Saving data...')
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])

        self.save_vertices(np.array(train_vertices), np.array(val_vertices), np.array(test_vertices))

def prepare_sliced_dataset(path):
    ComaDataset(path, pre_transform=Normalize())


def prepare_expression_dataset(path):
    test_exps = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up', 'mouth_down',
                 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
    for exp in test_exps:
        ComaDataset(path, split='expression', split_term=exp, pre_transform=Normalize())

def prepare_identity_dataset(path):
    test_ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170811_03274_TA',
                'FaceTalk_170904_00128_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170913_03279_TA',
                'FaceTalk_170728_03272_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170811_03275_TA',
                'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170915_00223_TA']

    for ids in test_ids:
        ComaDataset(path, split='identity', split_term=ids, pre_transform=Normalize())

def prepare_custom_dataset(path, test_set):
    ComaDataset(path, split='custom', split_term=test_set, pre_transform=Normalize())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('--dataset_dir', help='path where the npy dataset to be stored')
    parser.add_argument('-t', '--test_set', help='test sets')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    test_set = args.test_set
    if split == 'sliced':
        prepare_sliced_dataset(data_dir)
    elif split == 'expression':
        prepare_expression_dataset(data_dir)
    elif split == 'identity':
        prepare_identity_dataset(data_dir)
    elif split == 'custom':
        prepare_custom_dataset(data_dir, test_set)
    else:
        raise Exception("Only sliced, expression and identity split are supported")

