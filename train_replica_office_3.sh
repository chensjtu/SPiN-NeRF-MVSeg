export CUDA_VISIBLE_DEVICES=3

my_list=(3 8 11 14 15 18 19 25 29 30 32 33 38 39 43 51 54 55 61 65 72  76 78 82 87 91 95 96 101 111)

# 遍历列表并输出内容
for item in "${my_list[@]}"
do
    python DS_NeRF/run_nerf_replica.py --config DS_NeRF/configs/mv_config.txt \
                                        --render_factor 1 \
                                        --i_weight 4000 \
                                        --i_video 4000 \
                                        --N_iters 40001 \
                                        --expname office_3 \
                                        --datadir data/Replica/office_3/Sequence_1 \
                                        --factor 1 \
                                        --N_gt 0 \
                                        --dataset_type replica \
                                        --id_instance $item \
                                        --render_mask
done
