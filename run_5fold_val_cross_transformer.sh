
# Cross transformer/Learned Embedding/Learned Weighting

gpu_id=5

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type re \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_reconv_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type re \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_reconv_nAnI_affinity_fold1/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type re \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_reconv_nAnI_affinity_fold2/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type re \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_reconv_nAnI_affinity_fold3/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type re \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_reconv_nAnI_affinity_fold4/version_0"

# Cross transformer/AA+AA/Learned Weighting

gpu_id=5

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type aa2 \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_1e-3_L4_aa2conv_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type aa2 \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_1e-3_L4_aa2conv_nAnI_affinity_fold1/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type aa2 \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_1e-3_L4_aa2conv_nAnI_affinity_fold2/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type aa2 \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_1e-3_L4_aa2conv_nAnI_affinity_fold3/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type aa2 \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_1e-3_L4_aa2conv_nAnI_affinity_fold4/version_0"

# Cross transformer/AA+ESM/Learned Weighting

gpu_id=5

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type aa+esm \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type aa+esm \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold1/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type aa+esm \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold2/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type aa+esm \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold3/version_0"


python validate_affinity.py \
--gpu-id $gpu_id \
--model cross_transformer \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type aa+esm \
--pool_type conv \
--test_params_rootdir  "tensorboard/cross_trans_cos750_1.5k_5e-4_L4_esmYconv_nAnI_affinity_fold4/version_0"

