import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--no-progress', dest='progress', action='store_false')
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC(id_ec, ec_id, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion, scaler, amp_enabled, amp_dtype):
    model.train()
    total_loss = 0.
    start_time = time.time()

    loader = train_loader
    if args.progress:
        loader = tqdm(train_loader, desc=f'epoch {epoch}', leave=False)

    for batch, data in enumerate(loader):
        optimizer.zero_grad()
        anchor, positive, negative = data

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            anchor_out = model(anchor.to(device=device, dtype=dtype))
            positive_out = model(positive.to(device=device, dtype=dtype))
            negative_out = model(negative.to(device=device, dtype=dtype))
            loss = criterion(anchor_out, positive_out, negative_out)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)


def main():
    args = parse()
    seed_everything(args.seed)
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    amp_enabled = False
    amp_dtype = torch.float16
    scaler = None
    if args.precision != 'fp32':
        if not use_cuda:
            raise RuntimeError(f'--precision {args.precision} requires CUDA')
        amp_enabled = True
        if args.precision == 'bf16':
            if hasattr(torch.cuda, 'is_bf16_supported') and not torch.cuda.is_bf16_supported():
                raise RuntimeError('--precision bf16 requested but CUDA bf16 is not supported')
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, '| precision:', args.precision, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
 
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training
    epoch_iter = range(1, epochs + 1)
    if args.progress:
        epoch_iter = tqdm(epoch_iter, desc='training epochs')
    for epoch in epoch_iter:
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./data/model/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion,
                           scaler, amp_enabled, amp_dtype)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # remove tmp save weights
    for tmp_model in [
            './data/model/' + model_name + '.pth',
            './data/model/' + model_name + '_' + str(epoch) + '.pth']:
        if os.path.exists(tmp_model):
            os.remove(tmp_model)
    # save final weights
    torch.save(model.state_dict(), './data/model/' + model_name + '.pth')


if __name__ == '__main__':
    main()
