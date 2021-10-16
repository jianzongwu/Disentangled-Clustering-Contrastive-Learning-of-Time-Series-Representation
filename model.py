import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.encoder import OursEncoder
from models.triplet_loss import TripletLoss, TripletLossVaryingLength
import time
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

class OursModel:
    '''The Ours model '''

    def __init__(
        self,
        input_dims=1,
        output_dims=320,
        reduced_dims=160,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=10,
        classes=3,
        sim_fun='euclidean',
        cate_fun='argmax',
        save_path=None
    ):
        ''' Initialize our model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            classes (int): The number of encoders.
            k_loss_fun: (str): The k-means loss function.
            lamda: (float): The ratio of k-means loss.
            sim_fun: (str): The similarity function of clustering representations.
            cate_fun: (str): The categorical function of encoder choosing.

            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''

        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.classes = classes
        
        self.net = OursEncoder(input_dims=input_dims, output_dims=output_dims, reduced_dims=reduced_dims, hidden_dims=hidden_dims, depth=depth, sim_fun=sim_fun, cate_fun=cate_fun).to(self.device)
        
        self.n_epochs = 0
        self.n_iters = 0
        self.save_path = save_path
        
        self.loss = TripletLoss(
            nb_random_samples=10,
            negative_penalty=1,
        )
        self.loss_varying = TripletLossVaryingLength(
            nb_random_samples=10,
            negative_penalty=1,
        )
        
    def init_cores(self, x):
        features = self.encode(x, mode='class')
        repr_numpy = features.cpu().detach().numpy()
        km = KMeans(n_clusters=self.classes, random_state=0)
        km.fit(repr_numpy)
        centers = torch.from_numpy(km.cluster_centers_).to(self.device)
        self.net.cores = torch.nn.Parameter(centers)
        self.net.cores.requires_grad = False
        print("K-Means done")
        return km.labels_
        
    def print_cores(self, x):
        features = self.encode(x, mode='class')
        cores = torch.nn.functional.normalize(self.net.cores)
        features = torch.cat([features, cores], dim=0)
        features_numpy = features.cpu().detach().numpy()
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_result = tsne.fit_transform(features_numpy)
        print('t-SNE finished')
        print(tsne_result[-self.classes:])
        
    def fit(self, train_data, n_iters=2000, save_every=100, verbose=True, valid=False):
        varying = bool(np.isnan(np.sum(train_data)))
        temp = torch.from_numpy(train_data).unsqueeze(1).to(torch.float)
        train_data_gpu = temp.to(self.device)
        dataset = TensorDataset(temp)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        loss_log = {
            's': [],
            'k': []
        }
        iter = 0
        time_start = time.time()

        km_labels = self.init_cores(train_data)
        while True:
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                if not varying:
                    loss, k_loss = self.loss(x, self.net, train_data_gpu)
                else:
                    loss, k_loss = self.loss_varying(x, self.net, train_data_gpu)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_log['s'].append(loss.item())
                loss_log['k'].append(k_loss.item())
                iter += 1
                
                if verbose and iter % save_every == 0:
                    time_end = time.time()
                    print(f"Step #{iter}", int(time_end - time_start), "s")
                    print(f"Step #{iter}: avg_loss={sum(loss_log['s']) / iter}")
                    print(f"Step #{iter}: avg_loss_k={sum(loss_log['k']) / iter}")
                    if valid:
                        self.print_cores(train_data)
                        if os.path.exists(f'{self.save_path}/model_i{iter}.pth'):
                            print(f'{self.save_path}/model_i{iter}.pth ' + "already exists, skip.")
                        else:
                            self.save(f'{self.save_path}/model_i{iter}.pth')
                    time_start = time.time()
                if iter % n_iters == 0:
                    return loss_log
               
        return loss_log

    def encode(self, data, mode='result'):
        assert self.net is not None, 'please train or load a net first'
        varying = bool(np.isnan(np.sum(data)))
        org_training = self.net.training
        self.net.eval()
        dataset = TensorDataset(torch.from_numpy(data).unsqueeze(1).to(torch.float))
        loader = DataLoader(dataset, batch_size=50 if not varying else 1)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0].to(self.device)
                length = x.size(2) - torch.sum(torch.isnan(x[0, 0])).data.cpu().numpy()
                x = x[:, :, :length]
                if mode == 'result':
                    out = self.net(x, only_class=False)['repr']
                elif mode == 'cate':
                    out = self.net(x, only_class=True)['cates']
                elif mode == 'class':
                    out = torch.nn.functional.normalize(self.net.class_encoder(x), dim=1)
                elif mode == 'list':
                    out = torch.cat([self.net.encoder_list[i](x) for i in range(self.classes)], dim=1)
                output.append(out)
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output

    def save(self, fn):
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)