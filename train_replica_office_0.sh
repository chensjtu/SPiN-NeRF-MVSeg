export CUDA_VISIBLE_DEVICES=0

my_list=("3" "4" "7" "8" "9" "10" "12" "14" "17" "19" "21" "23" "26" "28" "29" "30" "36" "37" "40" "42" "44" "46" "54" "55" "57" "58" "61")

# 遍历列表并输出内容
for item in "${my_list[@]}"
do
    python DS_NeRF/run_nerf_replica.py --config DS_NeRF/configs/mv_config.txt \
                                        --render_factor 1 \
                                        --i_weight 4000 \
                                        --i_video 4000 \
                                        --N_iters 40001 \
                                        --expname office_0 \
                                        --datadir data/Replica/office_0/Sequence_1 \
                                        --factor 1 \
                                        --N_gt 0 \
                                        --dataset_type replica \
                                        --id_instance $item \
                                        --render_mask
done
