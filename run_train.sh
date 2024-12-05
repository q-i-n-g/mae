export CUDA_VISIBLE_DEVICES=0,1,2,6,7
export OMP_NUM_THREADS=1 
torchrun --nnodes=1 --nproc_per_node=5 main_pretrain.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 400 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.5 \
    --warmup_epochs 20 \
    --input_size 224 \
    --data_path Data/final_images \