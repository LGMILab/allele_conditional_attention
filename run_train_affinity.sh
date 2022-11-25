# CUDA_LAUNCH_BLOCKING=1


# python main_affinity.py \
# --gpu-id 2 \
# --logger tensorboard \
# --model bertlike \
# --data_filename mj_ranged_v1_0921.csv \
# --batch_size 1024 \
# --emb_type aa+esm \
# --pool_type conv \
# --fold 1 \
# --default_root_dir tensorboard/debug_cross_transformer

# --data_filename mj_ranged_v1_0921.csv \

# python main_affinity.py \
# --gpu-id 6 \
# --logger tensorboard \
# --model transformer \
# --data_filename mj_class_train.csv \
# --test_data_filename mj_class_valid.csv \
# --batch_size 1024 \
# --emb_type aa+esm \
# --pool_type conv \
# --fold 2 \
# --classes Positive-High Positive-Intermediate Positive-Low Negative \
# --default_root_dir ckpts/debug_transformer


# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --classes Positive-High Positive-Intermediate Positive-Low Negative \

# python main_affinity.py \
# --gpu-id 3 \
# --logger tensorboard \
# --model cnn \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --batch_size 1024 \
# --emb_type re \
# --fold 1 \
# --default_root_dir ckpts/debug_cnn_re

# python main_affinity.py \
# --gpu-id 3 \
# --mhc_class 2 \
# --logger tensorboard \
# --model transformer \
# --data_filename mhc_II_affinity_dataset_mut_5_fold.csv \
# --batch_size 1024 \
# --use_aa \
# --pool_type conv \
# --fold 1 \
# --default_root_dir ckpts/debug_transformer2_fold1

# python main_affinity.py \
# --gpu-id 5 \
# --logger tensorboard \
# --model gru \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --batch_size 512 \
# --use_aa \
# --pool_type average \
# --fold 1 \
# --default_root_dir ckpts/debug_gru_fold1


# python main_affinity.py \
# --gpu-id 5 \
# --logger tensorboard \
# --model transphla \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --batch_size 1024 \
# --fold 1 \
# --default_root_dir ckpts/debug_transphla_affinity_fold1

# python main_affinity.py \
# --gpu-id 7 \
# --logger tensorboard \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --batch_size 1024 \
# --fold 1 \
# --default_root_dir ckpts/debug_transformer_affinity_fold1

# python main_affinity.py \
# --gpu-id 7 \
# --logger tensorboard \
# --model cross_transformer \
# --data_filename mj_regress_curated_strat_5folds_0803.csv \
# --batch_size 1024 \
# --fold 1 \
# --default_root_dir ckpts/debug_cross_transformer_affinity_fold1

# python main_affinity.py \
# --gpu-id 4 \
# --logger tensorboard \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --fold 1 \
# --default_root_dir ckpts/debug_gru_affinity

# python main_affinity.py \
# --gpu-id 3 \
# --logger tensorboard \
# --model gru \
# --data_filename mj_regress_curated_5folds.csv \
# --teacher_weights_path ckpts/mj_gru_curated_v1_rank/gru/NEOAN-314/checkpoints/epoch=236-step=6635.ckpt \
# --distill_data_filename distillation_unlabeled_el_forba.csv \
# --distill_data_alpha .5 \
# --test_data_filename mj_regress_valid_curated_v1.csv \
# --default_root_dir ckpts/gru_distill_forba_hard

# python main_affinity.py \
# --gpu-id 8,9,10,11,12,13,14,15 \
# --logger tensorboard \
# --model gru \
# --data_filename distillation_unlabeled_el_forba.csv \
# --test_data_filename mj_regress_valid_curated_v1.csv \
# --teacher_weights_path ckpts/mj_gru_curated_v1_rank/gru/NEOAN-314/checkpoints/epoch=236-step=6635.ckpt \
# --default_root_dir ckpts/gru_distillation_el_forba


# python main_affinity.py \
# --gpu-id 12,13,14,15 \
# --logger tensorboard \
# --model gru \
# --data_filename mj_regress_train_curated_v1.csv \
# --test_data_filename mj_regress_valid_curated_v1.csv \
# --default_root_dir ckpts/mj_gru_curated_msa


# --data_filename mj_regress_train_ogvalue.csv \
# --test_data_filename mj_regress_valid_ogvalue.csv \
# --default_root_dir ckpts/mj_gru_reg_ogvalue

# --default_root_dir ckpts/mj_gru_reg

# --data_filename mhc_affinity_clean_1208_all_no_repeated_clamped_0_50k.csv \
# --fold 3 \

# --data_filename mj_regress_train.csv \
# --test_data_filename mj_regress_valid.csv \

# --data_filename mj_class_train_gmm_phase1.csv \
# --test_data_filename mj_class_valid_gmm_phase1.csv \
# --classes Positive-High Positive-Intermediate Positive-Low Negative \

# --data_filename mj_class_train_gmm_phase2.csv \
# --test_data_filename mj_class_valid_gmm_phase2.csv \
# --classes Positive-High Positive-Intermediate Positive-Low Negative \

# --data_filename mj_regress_train.csv \
# --test_data_filename mj_regress_valid.csv \

# --data_filename mj_og_regress_train.csv \
# --test_data_filename mj_og_regress_valid.csv \

# --data_filename mj_regress_train_mcdo1.csv \
# --test_data_filename mj_regress_valid.csv \

# --data_filename mj_class_train_gmm_phase2.csv \
# --test_data_filename mj_class_valid_gmm_phase2.csv \
# --classes Positive-High Positive-Intermediate Positive-Low Negative \

# --data_filename mhc_affinity_clean_1208_all_no_repeated_clamped_0_50k.csv \
# --fold 3 \
# --pretrained_weights_path ckpts/mj_gru_cls_phase2/gru/NEOAN-237/checkpoints/epoch=212-step=9584.ckpt \

# --data_filename mj_regress_train.csv \
# --test_data_filename mj_regress_valid.csv \