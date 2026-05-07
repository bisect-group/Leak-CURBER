import os
import torch as th
import torch.nn as nn

class KcatModel(nn.Module):
    def __init__(self, rate, device="cuda:0"):
        super(KcatModel, self).__init__()
        self.prot_norm = nn.BatchNorm1d(1024).to(device)
        #self.saprot_norm = nn.LayerNorm(446).to(device)
        self.molt5_norm = nn.BatchNorm1d(768).to(device)

        self.decoder = nn.Sequential(nn.Linear(1959, 256), nn.BatchNorm1d(256), nn.Dropout(p=rate), nn.ReLU(),).to(device)
        #self.attn = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Softmax()).to(device)

        self.out = nn.Sequential(nn.Linear(256, 1)).to(device)

    def forward(self, ezy_feats, sbt_feats):

        prot_feats = self.prot_norm(ezy_feats[:, :1024])
        molt5_feats = self.molt5_norm(sbt_feats[:, :768])
        macc_feats = sbt_feats[:, 768:]
        cplx_feats = th.cat([prot_feats, molt5_feats, macc_feats], axis=1)
        
        feats = self.decoder(cplx_feats) 
        out = self.out(feats)

        return out
        
