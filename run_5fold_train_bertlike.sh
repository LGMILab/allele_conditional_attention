
############
# Bertlike
############

code=main_affinity.py
logger=tensorboard
model=bertlike
batch_size=1024
data_filename=mj_regress_curated_strat_5folds_0803.csv
default_root_dir=tensorboard/bertlike_cos750_1.5k_1e-3_L4_esmYconv_nAnI_affinity_fold
emb_type="aa+esm"
pool_type="conv"

tmux new -s bertlike_esmYconv_affinity_5folds -d
tmux send-keys "conda activate allele_conditional" C-m
tmux send-keys "cd .." C-m
tmux send-keys "
python $code --emb_type $emb_type --pool_type $pool_type --gpu-id 6,7 --fold 0 --default_root_dir "${default_root_dir}0" --logger $logger --model $model --batch_size $batch_size --data_filename $data_filename &
python $code --emb_type $emb_type --pool_type $pool_type --gpu-id 8,9 --fold 1 --default_root_dir "${default_root_dir}1" --logger $logger --model $model --batch_size $batch_size --data_filename $data_filename &
python $code --emb_type $emb_type --pool_type $pool_type --gpu-id 10,11 --fold 2 --default_root_dir "${default_root_dir}2" --logger $logger --model $model --batch_size $batch_size --data_filename $data_filename &
python $code --emb_type $emb_type --pool_type $pool_type --gpu-id 12,13 --fold 3 --default_root_dir "${default_root_dir}3" --logger $logger --model $model --batch_size $batch_size --data_filename $data_filename &
python $code --emb_type $emb_type --pool_type $pool_type --gpu-id 14,15 --fold 4 --default_root_dir "${default_root_dir}4" --logger $logger --model $model --batch_size $batch_size --data_filename $data_filename &
wait" C-m