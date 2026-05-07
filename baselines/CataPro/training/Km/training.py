import pandas as pd
import numpy as np
import torch as th
from model import KcatModel
from torch.utils.data import DataLoader, Dataset
from utils import EarlyStopping, run_a_training_epoch, run_an_eval_epoch, write_logfile, \
    out_results, evaluate
from argparse import RawDescriptionHelpFormatter
import argparse
from sklearn.model_selection import KFold
import os
import datetime
import copy

class Mydatasets(Dataset):
    def __init__(self, values):
        self.values = values
    
    def __getitem__(self, idx):
        return self.values[idx]
    
    def __len__(self):
        return len(self.values)

class CVDatasets():
    def __init__(self, fpath, batch_size=32):
        self.batch_size = batch_size

        data_df = pd.read_pickle(fpath)
        self.data_index = data_df.index.tolist()
        data_values = data_df.values
        self.data_dict = dict(zip(self.data_index, data_values[:, :-1]))

        fold_indices = data_values[:, -1]
        self.split_index_dict = {}
        for i in range(10):
            valid_indices = []
            for fold, idx in zip(fold_indices, self.data_index):
                if fold == i:
                    valid_indices.append(idx)
            train_indices = [x for x in self.data_index if not x in valid_indices]
            self.split_index_dict[i] = [train_indices, valid_indices]

    def get_dataloader(self, idx):
        train_indices, valid_indices = self.split_index_dict[idx]
        print("train number:", len(train_indices))
        print("valid number:", len(valid_indices))

        train_values = th.from_numpy(np.concatenate([self.data_dict[k].reshape(1, -1) for k in train_indices], axis=0).astype(np.float32))
        valid_values = th.from_numpy(np.concatenate([self.data_dict[k].reshape(1, -1) for k in valid_indices], axis=0).astype(np.float32))

        train_datasets = Mydatasets(train_values)
        valid_datasets = Mydatasets(valid_values)

        train_dataloader = DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True, num_workers=8)
        valid_dataloader = DataLoader(valid_datasets, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_indices, valid_dataloader

if __name__ == "__main__":
    d = "Training ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-data_fpath", type=str, default="train_feats.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-lr", type=float, default=0.001,
                        help="Input. Learning rate.")
    parser.add_argument("-drop_rate", type=float, default=0.0,
                        help="Input. The rate of dropout.")
    parser.add_argument("-epochs", type=int, default=150,
                        help="Input. Epochs")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    args = parser.parse_args()

    data_fpath = args.data_fpath
    batch_size = args.batch_size
    lr = args.lr
    drop_rate = args.drop_rate
    epochs = args.epochs
    device = args.device
    patience = 20
    min_delta = 0.001

    cvdata = CVDatasets(fpath=data_fpath, batch_size=batch_size)
    all_pred_label = []
    all_indices = []
    for fold in range(10):
        train_dataloader, valid_keys, valid_dataloader = cvdata.get_dataloader(fold)
        model = KcatModel(rate=drop_rate, device=device)
        #optimizer = th.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = th.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

        stop = EarlyStopping(patience, min_delta)
        bestmodel = None
        
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_data = []
        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            train_eval = run_a_training_epoch(model, train_dataloader, optimizer, device)
            valid_pred, valid_label, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)

            record_data.append(np.concatenate([np.array([epoch]), train_eval, valid_eval], axis=0))
            write_logfile(epoch, record_data, "logfile_{}.csv".format(fold))
            is_bestmodel, stopping = stop.check(epoch, valid_eval[-1])

            if is_bestmodel == True:
                bestmodel = model.state_dict().copy()
                th.save(bestmodel, "{}_bestmodel.pth".format(fold))

            if stopping == True:
                print("Earlystopping !")
                break
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('time_running_{}.dat'.format(fold), 'w') as f:
            f.writelines('Start Time:  ' + start_time + '\n')
            f.writelines('End Time:  ' + end_time)

        model = KcatModel(rate=drop_rate, device=device)
        model.load_state_dict(th.load("{}_bestmodel.pth".format(fold), map_location=device))
        valid_pred, valid_label, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)
        out_results(valid_eval, "results_{}.csv".format(fold))
        
        all_indices += valid_keys
        all_pred_label.append(np.concatenate([valid_pred.reshape(-1, 1), valid_label.reshape(-1, 1)], axis=1))   
        
    all_pred_label = np.concatenate(all_pred_label, axis=0)
    final_df = pd.DataFrame(all_pred_label, index=all_indices, columns=["pred", "label"])
    final_df.to_csv("final_pred_label.csv", float_format="%.4f")

    _pcc, _scc, _r2, _rmse = evaluate(all_pred_label[:, 1], all_pred_label[:, 0])
    final_results = pd.DataFrame(np.array([[_pcc, _scc, _r2, _rmse]]), columns=["PCC", "SCC", "R2", "RMSE"])
    final_results.to_csv("final_results.csv")

