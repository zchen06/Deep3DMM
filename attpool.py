import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

from math import ceil, sqrt
import pdb

class AttPool(nn.Module):
    def __init__(self, config, const_k=None, const_q=None, prior_map=None):
        super(AttPool, self).__init__()
        self.num_k = config['num_k']
        self.num_q = config['num_q']
        self.init = 'constant'
        self.prior_init = config['prior_init'] if 'prior_init' in config else False
        self.prior_coef = config['prior_coef'] if 'prior_coef' in config else 0
        
        self.epsilon = 1e-12
        self.SCORE_MIN = 1e-25 
        self.SCORE_MAX = 1e25

        if 'dim_att' in config:
            if config['dim_att'] <= 0:
                self.dim_att = 1
            elif config['dim_att'] == ceil(config['dim_att']):
                self.dim_att = int(ceil(config['dim_att']))
            else:
                self.dim_att = int(ceil(self.num_q * config['dim_att']))
        else:
            self.dim_att = 3
            
        if 'top_k' in config and config['top_k'] < 0:
            self.top_k = 0
        elif 'top_k' in config and config['top_k'] < self.num_k:
            self.top_k = config['top_k'] if config['top_k']>=1 else config['top_k']*self.num_k
        else:
            self.top_k = 0
        
        self.key = Parameter(torch.Tensor(self.dim_att, self.num_k))
        self.query = Parameter(torch.Tensor(self.dim_att, self.num_q))
        self.weight_prior = Parameter(torch.Tensor(2, 1))
        self.const_k = const_k
        self.const_q = const_q
        self.prior_map = prior_map
        self.reset_parameters()

        self.norm = lambda x: F.normalize(x, p=1, dim=-1)


    def reset_parameters(self):        
        with torch.no_grad():
            self.weight_prior.copy_(torch.tensor([[self.prior_coef], [1.0-self.prior_coef]],
                                                 device=self.const_k.device))

        if self.init == 'uniform':
            init_1, init_2 = -0.1, 0.1
            nn.init.uniform_(self.key, a=init_1, b=init_2)
            nn.init.uniform_(self.query, a=init_1, b=init_2)
        elif self.init == 'normal':
            init_1, init_2 = -0.1, 0.1
            nn.init.normal_(self.key, mean=init_1, std=init_2)
            nn.init.normal_(self.query, mean=init_1, std=init_2)
        elif self.init == 'constant':
            with torch.no_grad():
                idx_dim_att = 0
                dim_const = self.const_k.size()[idx_dim_att]
                if dim_const < self.dim_att:
                    pad_dim = self.dim_att - dim_const
                    pad_zeros = torch.zeros([pad_dim, self.const_k.size()[1]], device=self.const_k.device)
                    const_k = torch.cat([self.const_k, pad_zeros], dim=idx_dim_att)
                    pad_zeros = torch.zeros([pad_dim, self.const_q.size()[1]], device=self.const_q.device)
                    const_q = torch.cat([self.const_q, pad_zeros], dim=idx_dim_att)
                elif dim_const == self.dim_att:
                    const_k = self.const_k
                    const_q = self.const_q
                elif dim_const > self.dim_att:
                    const_k = self.const_k[:self.dim_att,:]
                    const_q = self.const_q[:self.dim_att,:]                    

                const_kn = const_k + 0.1*torch.rand_like(const_k, device=const_k.device)
                const_qn = const_q + 0.1*torch.rand_like(const_q, device=const_q.device)
                self.key.copy_(const_kn)
                self.query.copy_(const_qn)
        else:
            raise NotImplementedError()

    def dist(self, key, query):
        key = torch.squeeze(key, -1)
        query = torch.squeeze(query, -1)
        norm_dim = -2
        key_normed = F.normalize(key, p=2, dim=norm_dim)
        query_normed = F.normalize(query, p=2, dim=norm_dim)
        dist = torch.transpose(torch.matmul(torch.transpose(key_normed, -2, -1), query_normed), -2, -1)
        return dist

    def get_score(self, key=None, query=None):        
        key_in = self.key if key is None else key
        query_in = self.query if query is None else query
            
        dist = self.dist(key_in, query_in)
        score = dist
        
        return score.clamp_(self.SCORE_MIN,self.SCORE_MAX)

    def get_topk(self, score=None, sort=True):
        if score is None:
            score = self.get_score()
        out_pts, in_pts = score.size()
        val, topk_index = torch.topk(score, self.top_k, dim=-1, sorted=sort)
        topk_val = self.norm(val)
        return topk_val, topk_index, out_pts, in_pts

    def gen_map(self, score):        
        if self.top_k == 0:
            topk_norm = self.norm(score)
        else:
            assert(self.top_k>0)
            val, idx = torch.topk(score, self.top_k, dim=-1, sorted=False)
            val_norm = self.norm(val)
            if len(score.size()) == 2:
                topk_norm = torch.zeros(self.num_q, self.num_k,
                                        device=self.key.device).scatter_(dim=-1, index=idx, src=val_norm)
            elif len(score.size()) == 3:
                topk_norm = torch.zeros(score.size()[0], self.num_q, self.num_k,
                                        device=self.key.device).scatter_(dim=-1, index=idx, src=val_norm)
            else:
                raise NotImplementedError()

        topk, top0 = topk_norm, self.norm(score)        
        if self.prior_map is not None and self.prior_init is True:
            topk, top0 = self.apply_prior_map(topk, top0)
        return topk, top0

    def apply_prior_map(self, topk, top0):
        denominator = torch.sum(self.weight_prior) \
            if torch.abs(torch.sum(self.weight_prior)) > self.epsilon else self.epsilon
        topk = torch.matmul(torch.cat([self.prior_map.unsqueeze(-1), topk.unsqueeze(-1)], dim=-1),
                            self.weight_prior).squeeze(-1) / denominator
        return topk, top0
    

    def forward(self, x, key=None, query=None):
        pmap, _ = self.gen_map(self.get_score(key=key, query=query))
        
        if len(pmap.size()) == 2:
            pmap = pmap.unsqueeze(0)
            out_feat = torch.matmul(pmap, x)
        elif len(pmap.size()) == 3:
            num_repeat = ceil(x.size()[-1]/pmap.size()[-1])
            num_map = pmap.size()[-1]
            out_feat = torch.cat([torch.matmul(pmap[:,:,idx_map],
                                               x[:,:,idx_map*num_repeat:(idx_map+1)*num_repeat])
                                  for idx_map in range(num_map)],dim=-1)
        else:
            raise NotImplementedError()
            
        return out_feat

    def extra_repr(self):
        return '{} -> {}, top {}, ch {}'.format(
            self.num_k, self.num_q, self.top_k, self.dim_att)
    

        
        
