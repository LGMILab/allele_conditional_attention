

# CNN/Learned Embedding/Learned Weighting
gpu_id=5
python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type re \
--test_params_rootdir  "tensorboard/cnn_re_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type re \
--test_params_rootdir  "tensorboard/cnn_re_nAnI_affinity_fold1/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type re \
--test_params_rootdir  "tensorboard/cnn_re_nAnI_affinity_fold2/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type re \
--test_params_rootdir  "tensorboard/cnn_re_nAnI_affinity_fold3/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type re \
--test_params_rootdir  "tensorboard/cnn_re_nAnI_affinity_fold4/version_0"

# CNN/AA+AA/Learned Weighting

gpu_id=5

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type aa2 \
--test_params_rootdir  "tensorboard/cnn_aa2_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type aa2 \
--test_params_rootdir  "tensorboard/cnn_aa2_nAnI_affinity_fold1/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type aa2 \
--test_params_rootdir  "tensorboard/cnn_aa2_nAnI_affinity_fold2/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type aa2 \
--test_params_rootdir  "tensorboard/cnn_aa2_nAnI_affinity_fold3/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type aa2 \
--test_params_rootdir  "tensorboard/cnn_aa2_nAnI_affinity_fold4/version_0"

# CNN/AA+ESM/Learned Weighting
gpu_id=5
python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 0 \
--emb_type aa+esm \
--test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold0/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 1 \
--emb_type aa+esm \
--test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold1/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 2 \
--emb_type aa+esm \
--test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold2/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 3 \
--emb_type aa+esm \
--test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold3/version_0"

python validate_affinity.py \
--gpu-id $gpu_id \
--model cnn \
--data_filename mj_regress_curated_strat_5folds_0803.csv \
--fold 4 \
--emb_type aa+esm \
--test_params_rootdir  "tensorboard/cnn_esmY_nAnI_affinity_fold4/version_0"

