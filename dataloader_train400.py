"""
    dataset and dataloader for train and eval
"""

import glob
import cv2
import os
import numpy as np
# from multiprocessing import Pool
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as pil_image
from tqdm import tqdm

# patch_size, stride = 40, 10
# aug_times = 1
# scales = [1, 0.9, 0.8, 0.7]
# batch_size = 128

#############################################################
# TRAIN DATASET RELATED
#############################################################

class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        clean = self.xs[index,0:1,:,:]
        noise = torch.randn(clean.size()).mul_(self.sigma / 255.0)
        noisy = clean + noise
        return noisy, clean

    def __len__(self):
        return self.xs.size(0)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name, patch_size, stride, scales, aug_times):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def data_generator(data_dir, patch_size, stride, scales, aug_times, batch_size, verbose=True):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    print('[DATASET] total {} train images'.format(len(file_list)))
    
    pbar = tqdm(range(len(file_list)), total=len(file_list))
    for i in pbar:
        patches = gen_patches(file_list[i], patch_size, stride, scales, aug_times)
        for patch in patches:
            data.append(patch)
        if verbose:
            pbar.set_postfix(complete=f"{i/len(file_list) * 100:.2f} %")
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch normalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^, numpy data size: ', data.shape)
    return data


#############################################################
# EVALUATION DATASET RELATED
#############################################################

def gen_eval_h5(images_dir, output_path, sigma):

    # image-wise evaluate with padding='same'
    h5_file = h5py.File(output_path, 'w')

    noisy_group = h5_file.create_group('noisy')
    clean_group = h5_file.create_group('clean')

    for imid, image_path in enumerate(sorted(glob.glob('{}/*'.format(images_dir)))):

        # load clean image
        print('[{}] dealing with {}'.format(imid, image_path))
        clean = pil_image.open(image_path).convert('L')
        w, h = clean.width, clean.height
        new_w, new_h = w - w % 8, h - h % 8
        clean = clean.resize((new_w, new_h), pil_image.ANTIALIAS)
        clean = np.array(clean).astype(np.float32)
        # add AWGN noise
        clean = clean / 255.0
        noise = np.random.randn(*clean.shape) * sigma / 255.0
        noisy = clean + noise

        noisy_group.create_dataset(str(imid), data=noisy)
        clean_group.create_dataset(str(imid), data=clean)

    h5_file.close()


def prepare_eval_h5file(config):

    os.makedirs('./eval_dataset_h5', exist_ok=True)
    #####################################
    SIGMA = config.sigma
    TEST_DIR = config.eval_dir
    OUTPUT_FILE = '{}_SIGMA{}.h5'.format(os.path.basename(TEST_DIR), SIGMA)
    #####################################

    h5file_path = os.path.join('./eval_dataset_h5', OUTPUT_FILE)
    if not os.path.exists(h5file_path):
        print('generate eval h5 to ', OUTPUT_FILE)
        gen_eval_h5(images_dir=TEST_DIR, \
                    output_path=os.path.join('./eval_dataset_h5', OUTPUT_FILE), \
                    sigma=SIGMA)
    else:
        print('{} already generated, use this file as eval dataset'.format(h5file_path))
    
    return h5file_path


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            noisy_img = np.expand_dims(f['noisy'][str(idx)][:,:], 0)
            clean_img = np.expand_dims(f['clean'][str(idx)][:,:], 0) 
            return noisy_img, clean_img

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['noisy'])

########################################################
############# train and eval dataloaders ###############
########################################################

def get_train_dataloader(config):
    print('[DATASET] generating patches')
    xs = data_generator(config.train_dir,
                        config.patch_size,
                        config.stride,
                        config.aug_scales,
                        config.aug_flip_times,
                        config.batch_size)
    xs = xs.astype('float32') / 255.0
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, [N, 1, H, W]
    print('[DATASET] begin build dataset by adding noise')
    train_dataset = DenoisingDataset(xs, config.sigma)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config.num_workers, \
                                    drop_last=True, batch_size=config.batch_size, shuffle=True)
    return train_dataloader, len(train_dataset)

def get_eval_dataloader(config):
    h5file_path = prepare_eval_h5file(config)
    eval_dataset = EvalDataset(h5file_path)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    return eval_dataloader, len(eval_dataset)


if __name__ == "__main__":
    from configs.config_ignet_sigma25 import get_config
    config = get_config()
    print('load train set')
    train_dataloader, len_trainset = get_train_dataloader(config)
    print('load eval set')
    eval_dataloader, len_evalset = get_eval_dataloader(config)
    
    print('test train set')
    for idx, data in enumerate(train_dataloader):
        if idx > 5:
            break
        print('{}th train batch'.format(idx))
        print(data[0].size(), data[1].size())
        #print(data[0], data[1])
    print('test eval set')
    for idx, data in enumerate(eval_dataloader):
        if idx > 5:
            break
        print('{}th eval batch'.format(idx))
        print(data[0].size(), data[1].size())

    print('all test passed, everything OK ... ')
