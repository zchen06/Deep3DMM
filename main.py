import argparse
import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset, ComaDataset_InMemory
from model import Coma, ComaAtt
from transform import Normalize

import pdb, ast, datetime
from math import ceil
from utils import scipy_to_torch_sparse

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('--viz', dest='visualize', action='store_true')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-p', '--checkpoint_dir', help='path where checkpoints file need to be stored')
    parser.add_argument('-m', '--modelname', default='Coma',
                        choices=['Coma', 'ComaAtt'], help='model name')
    parser.add_argument('--device_idx', type=int, help='cuda device index')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--num_threads', type=int, default=8, help='num_threads')
    parser.add_argument('--checkpoint_file', help='path of checkpoint file')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--eval', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('--rep_cudnn', default=False, action='store_true')
    parser.add_argument('--debug', type=ast.literal_eval, default=False, help='enable debug')    
    parser.add_argument('--epochs_eval', type=int, default=[-1], metavar='N', nargs='+', help='')
    parser.add_argument('--hier_matrices', default='downsampling_matrices', help='path of hierarchical matrices')
    parser.add_argument('--hier_matrices_save', help='path of hierarchical matrices')
    parser.add_argument('--Dataset_InMemory', type=ast.literal_eval, default=False)
    parser.add_argument('--num_workers', type=int, help='num_workers')

    
    parser.add_argument('--lr', dest='learning_rate', type=float)
    parser.add_argument('-z', '--dim', type=int, default=8, help='dimenstion of latent code')    
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--epoch', type=int, help='number of epochs')
        
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))


def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.num_threads)
    if args.rep_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    checkpoint_dir = os.path.join(checkpoint_dir,args.modelname)
    print(datetime.datetime.now())
    print('checkpoint_dir', checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']
        
    visualize = config['visualize'] if args.visualize is None else args.visualize
    output_dir = config['visual_output_dir']
    if output_dir:
        output_dir = os.path.join(output_dir, args.modelname)
    if visualize is True and not output_dir:
        print('No visual output directory is provided. \
        Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.train:
        eval_flag = True
    else:
        eval_flag = config['eval']
        
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread'] if args.num_workers is None else args.num_workers
    opt = config['optimizer']
    batch_size = config['batch_size'] if args.batch is None else args.batch
    val_losses, accs, durations = [], [], []

    if args.device_idx is None:
        device = torch.device("cuda:"+str(config['device_idx']) if torch.cuda.is_available() else "cpu")
    elif args.device_idx >= 0:
            device = torch.device("cuda:"+str(args.device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(config)

    ds_fname = os.path.join('./template/', data_dir.split('/')[-1] + '_'+args.hier_matrices+'.pkl')
    if not os.path.exists(ds_fname):
        print("Generating Transform Matrices ..")
        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
        with open(ds_fname, 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U}, fp)
    else:
        print("Loading Transform Matrices ..")
        with open(ds_fname, 'rb') as fp:
            downsampling_matrices = pickle.load(fp)
    
        M_verts_faces = downsampling_matrices['M_verts_faces']
        M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    nV_ref = []
    ref_mean = np.mean(M[0].v, axis=0)
    ref_std = np.std(M[0].v, axis=0)
    for i in range(len(M)):
        nv = 0.1 * (M[i].v - ref_mean)/ref_std
        nV_ref.append(nv)
        
    tV_ref = [torch.from_numpy(s).float().to(device) for s in nV_ref]
    
    print('Loading Dataset')

    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term,
                          pre_transform=normalize_transform)
    dataset_val = ComaDataset(data_dir, dtype='val', split=args.split, split_term=args.split_term,
                              pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term,
                               pre_transform=normalize_transform)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1 if visualize else batch_size, shuffle=False,
                             num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    if args.modelname in {'ComaAtt'}:
        gcn_model = eval(args.modelname)(dataset, config, D_t, U_t, A_t, num_nodes, tV_ref)
        gcn_params = gcn_model.parameters()
    else:
        gcn_model = eval(args.modelname)(dataset, config, D_t, U_t, A_t, num_nodes)
        gcn_params = gcn_model.parameters()
    
    params = sum(p.numel() for p in gcn_model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params)) 
    print(gcn_model)
    
    
    if opt == 'adam':
        optimizer = torch.optim.Adam(gcn_params, lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(gcn_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    if args.checkpoint_file:
        checkpoint_file = os.path.join(checkpoint_dir, str(args.checkpoint_file)+'.pt')
    else:
        checkpoint_file = config['checkpoint_file']
    if eval_flag and not checkpoint_file:
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')

    print(checkpoint_file)
    if checkpoint_file:
        print('Loading checkpoint file: {}.'.format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=device)
        start_epoch = checkpoint['epoch_num']
        gcn_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    gcn_model.to(device)
    
    if eval_flag:
        val_loss, euclidean_loss = evaluate(gcn_model, output_dir, test_loader,
                                            dataset_test, template_mesh, device, visualize)
        print('val loss', val_loss)
        print('euclidean error is {} mm'.format(1000*euclidean_loss))
        return

    best_val_loss = float('inf')
    val_loss_history = []
        
    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        train_loss = train(gcn_model, train_loader, len(dataset), optimizer, device)
        val_loss, _ = evaluate(gcn_model, output_dir, val_loader, dataset_val,
                               template_mesh, device, visualize=visualize)

        print('epoch {}, Train loss {:.8f}, Val loss {:.8f}'.format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            save_model(gcn_model, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)
            
        if epoch in args.epochs_eval or (val_loss <= best_val_loss and epoch > int(total_epochs*3/4)):
            val_loss, euclidean_loss = evaluate(gcn_model, output_dir, test_loader,
                                                dataset_test, template_mesh, device, visualize)
            print('epoch {} with val loss {}'.format(epoch, val_loss))
            print('euclidean error is {} mm'.format(1000*euclidean_loss))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        


def train(gcn_model, train_loader, len_dataset, optimizer, device):
    gcn_model.train()
    total_loss = 0
    total_rec_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = gcn_model(data)
        loss_data = F.l1_loss(out, data.y)
        loss =  loss_data
        total_loss += data.num_graphs * loss.item()
        total_rec_loss += data.num_graphs * loss_data.item()
        loss.backward()
        optimizer.step()
        
    print('total loss: {:.5f}, data loss: {:.5f}'.format(total_loss/len_dataset,total_rec_loss/len_dataset))
    return total_loss / len_dataset


def evaluate(gcn_model, output_dir, test_loader, dataset, template_mesh, device, visualize=False):
    gcn_model.eval()
    total_loss = 0
    total_euclidean_loss = 0
    euclidean_loss_list = []

    std = dataset.std.unsqueeze(0).numpy()
    mean = dataset.mean.unsqueeze(0).numpy()
    
    if visualize:
        meshviewer = MeshViewers(shape=(1, 2))
        size = test_loader.__len__()
        predictions = [0]*size
        gts = [0]*size
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = gcn_model(data)
        
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()
        
        save_out = out.reshape(data.num_graphs,-1,3).detach().cpu().numpy()
        save_out = save_out*std + mean
        expected_out = (data.y.reshape(data.num_graphs,-1,3).detach().cpu().numpy())*std + mean
        euclidean_loss = np.mean(np.sqrt(np.sum((save_out-expected_out)**2, axis=-1)))
        euclidean_loss_list.append(np.sqrt(np.sum((save_out-expected_out)**2, axis=-1)))
        total_euclidean_loss += data.num_graphs * euclidean_loss

        if visualize:
            predictions[i] = save_out
            gts[i] = expected_out
        if visualize and i % 100 == 0:
            result_mesh = Mesh(v=save_out, f=template_mesh.f)
            expected_mesh = Mesh(v=expected_out, f=template_mesh.f)
            meshviewer[0][0].set_dynamic_meshes([result_mesh])
            meshviewer[0][1].set_dynamic_meshes([expected_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(output_dir, 'file'+str(i)+'.png'), blocking=False)

    eu_list = 1000*np.concatenate(euclidean_loss_list, axis=0)
    print('Euclidean mean: {:.5f}, std: {:.5f}, median: {:.5f} mm'.format(np.mean(eu_list), np.std(eu_list), np.mean(eu_list)))
    return total_loss/len(dataset), total_euclidean_loss/len(dataset)
    
    
if __name__ == '__main__':
    args = get_args()
    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
