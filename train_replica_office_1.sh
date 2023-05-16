export CUDA_VISIBLE_DEVICES=1

my_list=(3 7 9 11 13 14 15 17 23 24 29 32 33 36 37 39 42 44 45 46)

# 遍历列表并输出内容
for item in "${my_list[@]}"
do
    python DS_NeRF/run_nerf_replica.py --config DS_NeRF/configs/mv_config.txt \
                                        --render_factor 1 \
                                        --i_weight 4000 \
                                        --i_video 4000 \
                                        --N_iters 40001 \
                                        --expname office_1 \
                                        --datadir data/Replica/office_1/Sequence_1 \
                                        --factor 1 \
                                        --N_gt 0 \
                                        --dataset_type replica \
                                        --id_instance $item \
                                        --render_mask
done
