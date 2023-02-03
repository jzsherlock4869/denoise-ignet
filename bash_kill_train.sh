ps -ef | grep train_ignet_denoise | awk -F" " '{print $2}' | xargs kill -9
echo "killed training process, remaining : "
ps -ef | grep train