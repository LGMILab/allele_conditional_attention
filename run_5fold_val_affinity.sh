
# transformer

gpu_id=5

python validate_affinity.py \
--gpu-id $gpu_id \
--model transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--emb_type aa+esm \
--pool_type conv \
--fold 0 \
--test_params_rootdir  "tensorboard/transformer_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--emb_type aa+esm \
--pool_type conv \
--fold 1 \
--test_params_rootdir  "tensorboard/transformer_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold1/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--emb_type aa+esm \
--pool_type conv \
--fold 2 \
--test_params_rootdir  "tensorboard/transformer_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold2/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--emb_type aa+esm \
--pool_type conv \
--fold 3 \
--test_params_rootdir  "tensorboard/transformer_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold3/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--emb_type aa+esm \
--pool_type conv \
--fold 4 \
--test_params_rootdir  "tensorboard/transformer_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold4/version_3"

# # bertlike

# gpu_id=5

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model bertlike \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 0 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/bertlike_cos750_1.5k_5e-4_L4_esmYconv_yAyI_noineq_affinity_fold0/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model bertlike \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 1 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/bertlike_cos750_1.5k_5e-4_L4_esmYconv_yAyI_noineq_affinity_fold1/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model bertlike \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 2 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/bertlike_cos750_1.5k_5e-4_L4_esmYconv_yAyI_noineq_affinity_fold2/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model bertlike \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 3 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/bertlike_cos750_1.5k_5e-4_L4_esmYconv_yAyI_noineq_affinity_fold3/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model bertlike \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 4 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/bertlike_cos750_1.5k_5e-4_L4_esmYconv_yAyI_noineq_affinity_fold4/version_0"

# # cross transformer

# gpu_id=5

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 0 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 1 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold1/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 2 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold2/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 3 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold3/version_0"


# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_noineq_5folds_0927.csv \
# --fold 4 \
# --emb_type aa+esm \
# --pool_type conv \
# --test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold4/version_0"


# # gru
# gpu_id=5
# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 0 \
# --emb_type aa2 \
# --test_params_rootdir  "tensorboard/gru_L4_aa2_nAnI_affinity_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 1 \
# --emb_type aa2 \
# --test_params_rootdir  "tensorboard/gru_L4_aa2_nAnI_affinity_fold1/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 2 \
# --emb_type aa2 \
# --test_params_rootdir  "tensorboard/gru_L4_aa2_nAnI_affinity_fold2/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 3 \
# --emb_type aa2 \
# --test_params_rootdir  "tensorboard/gru_L4_aa2_nAnI_affinity_fold3/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 4 \
# --emb_type aa2 \
# --test_params_rootdir  "tensorboard/gru_L4_aa2_nAnI_affinity_fold4/version_0"


# # gru
# gpu_id=5
# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 0 \
# --test_params_rootdir  "tensorboard/gru_affinity_swa5e-5_fold0/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 1 \
# --test_params_rootdir  "tensorboard/gru_affinity_swa5e-5_fold1/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold1/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 2 \
# --test_params_rootdir  "tensorboard/gru_affinity_swa5e-5_fold2/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold2/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 3 \
# --test_params_rootdir  "tensorboard/gru_affinity_swa5e-5_fold3/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold3/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 4 \
# --test_params_rootdir  "tensorboard/gru_affinity_swa5e-5_fold4/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold4/version_0"


# # cnn
# gpu_id=5
# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 0 \
# --emb_type aa+esm \
# --test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold0/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 1 \
# --emb_type aa+esm \
# --test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold1/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold1/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 2 \
# --emb_type aa+esm \
# --test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold2/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold2/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 3 \
# --emb_type aa+esm \
# --test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold3/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold3/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --fold 4 \
# --emb_type aa+esm \
# --test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold4/version_0"
# # --test_params_rootdir  "tensorboard/gru_affinity_swa1e-4_fold4/version_0"


# # transformer

# # --test_params_rootdir  "tensorboard/preln_cos.5k_affinity_fold0/version_0"


# gpu_id=10
# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 0 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa5e-5_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 1 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa5e-5_fold1/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 2 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa5e-5_fold2/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 3 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa5e-5_fold3/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 4 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa5e-5_fold4/version_0"


# gpu_id=10
# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 0 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa1e-4_fold0/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 1 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa1e-4_fold1/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 2 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa1e-4_fold2/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 3 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa1e-4_fold3/version_0"

# python validate_affinity.py \
# --gpu-id $gpu_id \
# --model transformer \
# --pre_ln True \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 4 \
# --test_params_rootdir  "tensorboard/preln_cos.1k_affinity_swa1e-4_fold4/version_0"




