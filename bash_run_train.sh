mkdir ./logs
CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignet --sigma 15 > logs/tr_ignet_sigma15.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignet --sigma 25 > logs/tr_ignet_sigma25.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignet --sigma 50 > logs/tr_ignet_sigma25.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignetp --sigma 15 > logs/tr_ignetp_sigma15.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignetp --sigma 25 > logs/tr_ignetp_sigma25.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_ignet_denoise.py --arch ignetp --sigma 50 > logs/tr_ignetp_sigma25.log 2>&1 &