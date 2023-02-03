
import importlib
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR 
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from archs.ignet import IGNet
from dataloader_train400 import get_train_dataloader, get_eval_dataloader
from utils import AverageMeter, calc_psnr
from visualize import vis_model

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def start_train(config):

    print("=== start training ===")
    print(config.print_all())
    config.output_dir = os.path.join(config.output_dir, 'sigma{}'.format(config.sigma), config.model_arch)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)

    # select model
    model = IGNet(**config.model_params).to(device)

    if len(config.use_gpus) > 1:
        model = nn.DataParallel(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, config.num_epochs, eta_min=1e-5) 

    print('load train set || ', get_now())
    train_dataloader, len_trainset = get_train_dataloader(config)
    # train_dataloader, len_trainset = get_train_dataloader_h5(config)
    print('{} train patches loaded || {}'.format(len_trainset, get_now()))
    print('load eval set')
    eval_dataloader, len_evalset = get_eval_dataloader(config)
    print('{} eval images loaded'.format(len_evalset))

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # merely test the vis_model function, comment when actually run
    # print("start visualize in Epoch {}".format(-1))
    # vis_model(model, eval_dataloader, config, -1)

    for epoch in range(config.num_epochs):
        
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len_trainset) as t:
            t.set_description('epoch: {}/{}'.format(epoch, config.num_epochs - 1))
        
            for data in train_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
                
                preds, _ = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        scheduler.step()

        model.eval()
        torch.save(model.state_dict(), os.path.join(config.output_dir, 'epoch_{}.pth'.format(epoch)))
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data
            # print(inputs.size(), labels.size())
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)

            with torch.no_grad():
                preds = model(inputs)[0].clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), 1)
        #print('eval psnr (epoch {}): {:.2f} || timestamp {}'.format(epoch, epoch_psnr.avg, get_now()))
        cur_lr = scheduler.get_last_lr()[0]

        if epoch % config.vis_interval == 0:
            print("start visualize in Epoch {}".format(epoch))
            vis_model(model, eval_dataloader, config, epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print('eval psnr (epoch {}): {:.2f}dB [cur best {:.2f}dB, cur lr {:.8f}] || timestamp {}'.format(epoch, epoch_psnr.avg, best_psnr, cur_lr, get_now()))
    
    print("=========== training finished ==============")
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(config.output_dir, 'best.pth'))


if __name__ == "__main__":

    # parse argument
    parser = argparse.ArgumentParser(description='input the noise level (sigma) 15, 30 or 50')
    parser.add_argument('--sigma', type=int, required=True, help='noise level, one of 15, 30, 50')
    parser.add_argument('--arch', type=str, required=True, help='network arch, ignet or ignetp')
    args = parser.parse_args()

    opt = getattr(importlib.import_module(f'configs.config_{args.arch}_sigma{args.sigma}'), 'get_config')()
    start_train(opt)
