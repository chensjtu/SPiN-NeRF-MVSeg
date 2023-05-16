export CUDA_VISIBLE_DEVICES=2

my_list=(0 2 8 9 13 14 17 19 23 27 40 41 47 49 51 54 58 60 65 67 70 71 72 73 78 85 90 92 93)

# 遍历列表并输出内容
for item in "${my_list[@]}"
do
    python DS_NeRF/run_nerf_replica.py --config DS_NeRF/configs/mv_config.txt \
                                        --render_factor 1 \
                                        --i_weight 4000 \
                                        --i_video 4000 \
                                        --N_iters 40001 \
                                        --expname office_2 \
                                        --datadir data/Replica/office_2/Sequence_1 \
                                        --factor 1 \
                                        --N_gt 0 \
                                        --dataset_type replica \
                                        --id_instance $item \
                                        --render_mask
done
