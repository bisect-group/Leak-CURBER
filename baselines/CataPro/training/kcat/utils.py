import os
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

def PCC(y_true, y_pred):
    fsp = y_pred - np.mean(y_pred)
    fst = y_true - np.mean(y_true)

    devP = np.std(y_pred)
    devT = np.std(y_true)

    return np.mean(fsp * fst) / (devP * devT)

def SCC(y_true, y_pred):
    X_rank = rankdata(y_pred)
    Y_rank = rankdata(y_true)

    return PCC(Y_rank, X_rank)

def R_2(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - y_true_mean))

    return 1 - numerator/denominator

def evaluate(y_true, y_pred):
    _pcc = PCC(y_true, y_pred)
    _rmse = RMSE(y_true, y_pred)
    _r2 = R_2(y_true, y_pred)
    _scc = SCC(y_true, y_pred)

    return _pcc, _scc, _r2, _rmse

def out_results(values, file_):
    columns = ["valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss"]
    df = pd.DataFrame(values.reshape(1, -1), columns=columns)
    df.to_csv(file_, float_format="%.5f")

def write_logfile(epoch, record_data, logfile):
    if epoch == 0:
        if os.path.exists(logfile):
            os.remove(logfile)

    index = [x for x in range(epoch + 1)]
    values = np.array(record_data).reshape(epoch+1, -1)

    columns = ["epoch", "train_pcc", "train_scc", "train_r2", "train_rmse", "train_loss", 
                        "valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss"]
    df = pd.DataFrame(values, index=index, columns=columns)
    df.to_csv(logfile, float_format="%.4f")


class EarlyStopping(object):
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = 0
        self.count_epoch = 0
        self.stop = False
        self.is_bestmodel = False

    def check(self, epoch, cur_loss):
        if epoch == 0:
            self.min_loss = cur_loss
            self.count_epoch += 1
            self.is_bestmodel = True
        else:
            if cur_loss < self.min_loss - self.min_delta:
                self.min_loss = cur_loss
                self.count_epoch = 0
                self.is_bestmodel = True
            else:
                self.count_epoch += 1
                self.is_bestmodel = False

        if self.count_epoch == self.patience:
            self.stop = True

        return self.is_bestmodel, self.stop

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, y_true, y_pred):
        loss = th.sqrt(self.mse(y_true, y_pred) + self.eps)
        return loss
rmse_loss = RMSELoss()
    
def run_a_training_epoch(model, data_loader, optimizer, device="cuda:0"):
    model.train()

    total_loss = 0
    y_label = []
    y_pred = []
    for step, data in enumerate(data_loader):
        data = data.to(device)
        ezy_feats = data[:, :1024]
        sbt_feats = data[:, 1024:-1]
        label = data[:, -1]

        pred_kcat = model(ezy_feats, sbt_feats)
        loss = rmse_loss(label, pred_kcat.ravel())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        th.cuda.empty_cache()

        with th.no_grad():
            total_loss += loss.cpu().detach().numpy()
            y_label.append(label.cpu().detach().numpy().ravel())
            y_pred.append(pred_kcat.cpu().detach().numpy().ravel())

    y_pred = np.concatenate(y_pred, axis=0)
    y_label = np.concatenate(y_label, axis=0)

    _pcc, _scc, _r2, _rmse = evaluate(y_label, y_pred)
    N = len(data_loader)

    return np.array([_pcc, _scc, _r2, _rmse, total_loss/N])

def run_an_eval_epoch(model, data_loader, device="cuda"):
    model.eval()

    with th.no_grad():
        total_loss = 0
        y_label = []
        y_pred = []
        for step, data in enumerate(data_loader):
            data = data.to(device)
            ezy_feats = data[:, :1024]
            sbt_feats = data[:, 1024:-1]
            label = data[:, -1]
            
            pred_kcat = model(ezy_feats, sbt_feats)
            #loss = mse_loss(label, pred_kcat.ravel())
            loss = rmse_loss(label, pred_kcat.ravel())

            total_loss += loss.cpu().detach().numpy()
            y_label.append(label.cpu().detach().numpy().ravel())
            y_pred.append(pred_kcat.cpu().detach().numpy().ravel())

        y_pred = np.concatenate(y_pred, axis=0)
        y_label = np.concatenate(y_label, axis=0)

        _pcc, _scc, _r2, _rmse = evaluate(y_label, y_pred)
        N = len(data_loader)

        return y_pred, y_label, np.array([_pcc, _scc, _r2, _rmse, total_loss/N])
