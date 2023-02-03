import torch
from utils import calc_psnr
import cv2
import os
import numpy as np

def concat_imgs(img_ls):
    """
    img size : [h, w, c]
    """
    for rid, imgs in enumerate(img_ls):
        for cid, img in enumerate(imgs):
            if cid == 0:
                row_cat = img
            else:
                row_cat = np.concatenate((row_cat, img), axis=1)
        if rid == 0:
            tot_cat = row_cat
        else:
            tot_cat = np.concatenate((tot_cat, img), axis=0)
    return tot_cat


def vis_model(model, eval_dataloader, opt, epoch_id):
    if not os.path.exists(os.path.join(opt.vis_dir, str(epoch_id))):
        os.makedirs(opt.vis_dir, exist_ok=True)
    print('visualize and save in {}'.format(opt.vis_dir))
    device = 'cuda'
    for idx, data in enumerate(eval_dataloader):
        inputs, labels = data   # [1, 1, w, h], [1, 1, w, h]
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            preds, _ = model(inputs)
            preds = preds.clamp(0.0, 1.0)
        cur_psnr = calc_psnr(preds, labels)
        
        im_inputs = np.uint8(inputs[0,0,:,:].clamp(0.0, 1.0).cpu() * 255)
        im_labels = np.uint8(labels[0,0,:,:].cpu() * 255)
        im_preds = np.uint8(preds[0,0,:,:].cpu() * 255)
        img_ls = [[im_inputs, im_labels, im_preds]]
        vis_im = concat_imgs(img_ls)
        os.makedirs(os.path.join(opt.vis_dir, 'epoch_{}'.format(epoch_id)), exist_ok=True)
        cv2.imwrite(os.path.join(opt.vis_dir, 'epoch_{}'.format(epoch_id), 'img{}_psnr{:.4f}.png'.format(idx, cur_psnr)), vis_im)
        print('save for image {} in epoch {}, with psnr {:.4f}'.format(idx, epoch_id, cur_psnr))
    return

