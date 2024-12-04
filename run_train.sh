export CUDA_VISIBLE_DEVICES=2,3,4
export OMP_NUM_THREADS=1 
torchrun --nnodes=1 --nproc_per_node=3 main_pretrain.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 200 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.5 \
    --warmup_epochs 10 \
    --input_size 224 \
    --data_path Data/combined_images