###################### usage ######################
# start train: pssh -i -t 0 -h cur.hosts bash InstantID/train_instantId_sdxl.sh
# kill process: pssh -i -t 0 -h cur.hosts 'ps -ef|grep "train_instantId_sdxl.py"|grep -v grep|cut -c 9-16|xargs kill -9'

# source /apdcephfs_cq8/share_1367250/francofhzhu/anaconda3/bin/activate
# conda activate cog

cd InstantID

# deepspeed config
export DS_SKIP_CUDA_CHECK=1
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=1

# 太极平台多机多卡设置，网络通信
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_DEBUG=INFO # DEBUG打印日志的等级
# huggingface config
export HF_HOME=/apdcephfs_cq8/share_1367250/francofhzhu/huggingface
# SDXL Model
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MODEL_NAME=${HF_HOME}/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
# CLIP Model
export ENCODER_NAME="IP-Adapter/sdxl_models/image_encoder"
# pretrained InstantID model
export ADAPTOR_NAME="./checkpoints/ip-adapter.bin"
export CONTROLNET_NAME="./checkpoints/ControlNetModel"
# export UNET_NAME="./checkpoints/sdxl_lightning_4step_unet.safetensors"

# Dataset
export ROOT_DATA_DIR="/"
# This json file ' format:
# {"file_name": "/data/train_data/images_part0/84634599103.jpg", "additional_feature": "myolv1,a man with glasses and a
# tie on posing for a picture in front of a window with a building in the background, Andrew Law, johnson ting, a picture,
# mannerism", "bbox": [-31.329412311315536, 160.6865997314453, 496.19240215420723, 688.1674156188965],
# "landmarks": [[133.046875, 318], [319.3125, 318], [221.0625, 422], [153.515625, 535], [298.84375, 537]],
# "insightface_feature_file": "/data/feature_data/images_part0/84634599103.bin"}
export JSON_FILE="./portrait_train_evaclip/merge_train_data.txt"


# Output
export OUTPUT_DIR="InstantID_output/v1"
mkdir -p $OUTPUT_DIR

echo "NODE_NUM $NODE_NUM"
echo "TAIJI_HOST_NUM $TAIJI_HOST_NUM"
echo "OUTPUT_DIR: $OUTPUT_DIR"
# one machine multi gpu
# accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \

# multi machine multi gpu
accelerate launch --machine_rank $INDEX \
  --main_process_ip $CHIEF_IP \
  --main_process_port 29500 \
  --num_processes $NODE_NUM \
  --num_machines $TAIJI_HOST_NUM \
  train_instantId_sdxl.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --controlnet_model_name_or_path $CONTROLNET_NAME \
  --image_encoder_path $ENCODER_NAME \
  --pretrained_ip_adapter_path $ADAPTOR_NAME \
  --data_root_path $ROOT_DATA_DIR \
  --data_json_file $JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --clip_proc_mode orig_crop \
  --mixed_precision bf16 \
  --resolution 1024 \
  --learning_rate 1e-5 \
  --weight_decay=0.01 \
  --num_train_epochs 2 \
  --train_batch_size 2 \
  --max_data_loader_n_workers 8 \
  --checkpoints_total_limit 6 \
  --save_steps 5000 \
  --num_inference_steps 3 \
  --id_loss_weight 10 \
  --id_loss_warmup_steps 100000 \
  --inference_guidance_scale 1.0 \
  --deepspeed \
  --zero_stage 1 \
  > ${OUTPUT_DIR}/train_log_node${INDEX}.txt 2>&1


