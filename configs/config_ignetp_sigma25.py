"""
    config file for ignet+ (larger than base in num of feature maps)
"""

import os

class Config():
    def __init__(self) -> None:

        self.seed = 2021

        #### dataset setting
        self.train_dir = '../datasets/MWCNN_Trainset_YChannel'
        self.eval_dir = '../datasets/testsets/BSD68'
        self.patch_size = 128
        self.stride = 128
        self.sigma = 25
        self.aug_flip_times = 2
        self.aug_scales = [1, 0.9, 0.8, 0.7]
        self.num_workers = 8
        
        #### train setting
        self.use_gpus = [0, 1, 2, 3]
        self.model_arch = 'ignetp'
        self.model_params = {
            "in_ch": 1,
            "out_ch": 1,
            "base_ft": 64,
            "mid_ft": 32
        }
        self.output_dir = '../output'
        self.batch_size = 512
        self.num_epochs = 100
        self.lr = 5e-4
       
        #### eval setting
        self.vis_dir = '../pred_vis_large'
        self.vis_interval = 1

        self.exp_name = 'arch_{}_bs{}_lr{}_p{}_s{}_sigma{}_augf{}_augs{}'\
            .format(self.model_arch, self.batch_size, self.lr, self.patch_size, \
                self.stride, self.sigma, self.aug_flip_times, len(self.aug_scales))

        # self.train_h5_path = '../datasets/train_h5/{}/MWCNN_Trainset_YChannel.h5'.format(self.exp_name)
        self.vis_dir = os.path.join(self.vis_dir, self.exp_name)

    def print_all(self):
        for k, v in self.__dict__.items():
            print("[CONFIG]    set {} -> {}".format(k, v))


def get_config():
    configure = Config()
    return configure
