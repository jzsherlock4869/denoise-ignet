import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

try:
    from .net_utils.wavelet import DWT, IWT
    from .net_utils.dncnn import DnCNNEncoder
except:
    from net_utils.wavelet import DWT, IWT
    from net_utils.dncnn import DnCNNEncoder

class GuidedRefineBlock(nn.Module):
    
    def __init__(self, in_ch_1, in_ch_2, mid_ch, out_ch):
        super(GuidedRefineBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        
        self.use_res = (in_ch_2 == out_ch)
        self.conv1 = nn.Conv2d(in_ch_1, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(in_ch_2, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv_fuse = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.conv_last = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x_1, x_2):
        f_1 = self.conv1(x_1)
        feat_1 = self.act(self.bn1(f_1))
        f_2 = self.conv2(x_2)
        feat_2 = self.act(self.bn2(f_2))
        fused = self.conv_fuse(torch.cat([feat_1, feat_2], dim=1))
        feat_fused = self.act(self.bn3(fused))
        output = self.act(self.conv_last(feat_fused))
        if self.use_res:
            output = output + x_2
        return output


class IGNet(nn.Module):
    """
        main structure of progressive low-frequency guided network
    """
    def __init__(self, in_ch=1, out_ch=1, base_ft=64, mid_ft=16, print_network=False):
        super(IGNet, self).__init__()

        ft = base_ft
        mch_0, mch_1, mch_2 = mid_ft * 4, mid_ft * 2, mid_ft

        # TODO: retrain ignet base with ignet+'s settings
        if base_ft == 32:
            self.type = 'base'
        else:
            self.type = 'large'

        #============#
        #   ENCODER  #
        #============#
        self.dwt_split = DWT()
        self.dwt_merge = IWT()
        self.feat_extractor = DnCNNEncoder(depth=5, n_ft=base_ft, in_chs=in_ch, out_ft=base_ft, res=False)
        #============#
        #   REFINER  #
        #============#
        # nested conv operations
        # mid_ch controls the model size
        # 1st level
        self.conv0_0 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_0, out_ch = 3*ft)
        self.conv0_1 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_0, out_ch = 3*ft)
        self.conv0_2 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_0, out_ch = 3*ft)
        # 2nd level
        self.conv1_0 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_1, out_ch = 3*ft)
        self.conv1_1 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_1, out_ch = 3*ft)
        # 3rd level
        self.conv2_0 = GuidedRefineBlock(in_ch_1 = ft, in_ch_2 = 3 * ft, mid_ch=mch_2, out_ch = 3*ft)
        # low-frequency refiner
        self.low_refiner = nn.Conv2d(ft, ft, kernel_size=3, padding=1, bias=True)
        #============#
        #   DECODER  #
        #============#
        self.final_refiner = DnCNNEncoder(depth=5, n_ft=base_ft, in_chs=ft, out_ft=out_ch, res=True)

        if print_network:
            print(self.__dict__)


    def forward(self, x):
        feat = self.feat_extractor(x)

        lb_0, hb_0 = self.dwt_split(feat)    # 1/2, 1/2 
        r0_0 = self.conv0_0(lb_0, hb_0)      # 1/2

        lb_1, hb_1 = self.dwt_split(lb_0)    # 1/4, 1/4
        r1_0 = self.conv1_0(lb_1, hb_1)      # 1/4

        r0_1 = self.conv0_1(self.dwt_merge(lb_1, r1_0), r0_0)

        lb_2, hb_2 = self.dwt_split(lb_1)

        if self.type == 'base':
            # ignet base is trained with following settings
            lb_2_refined = self.low_refiner(lb_2)
            r2_0 = self.conv2_0(lb_2_refined, hb_2)
            out_feat_2 = self.dwt_merge(lb_2, r2_0)
        else:
            # ignet+ and ignet++ is trained with following settings
            lb_2_refined = self.low_refiner(lb_2) + lb_2
            r2_0 = self.conv2_0(lb_2_refined, hb_2)
            out_feat_2 = self.dwt_merge(lb_2_refined, r2_0)

        r1_1 = self.conv1_1(out_feat_2, r1_0)
        out_feat_1 = self.dwt_merge(out_feat_2, r1_1)
        r0_2 = self.conv0_2(out_feat_1, r0_1)
        out_feat_0 = self.dwt_merge(out_feat_1, r0_2)

        output = self.final_refiner(out_feat_0)
        aux_outs = (out_feat_0, out_feat_1, out_feat_2)

        return output, aux_outs


if __name__ == "__main__":
    device = 'cuda'
    ignet = IGNet(in_ch=1, out_ch=1, device=device).to(device)
    print(ignet)
    dummy_input = torch.randn(4, 1, 64, 64).to(device)
    output, aux_out = ignet(dummy_input)
    print(dummy_input.shape, output.shape, [i.shape for i in aux_out])
