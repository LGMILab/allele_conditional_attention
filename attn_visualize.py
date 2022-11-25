import pytorch_lightning as pl
import torch

from lightning_modules import AffinityPL
from dataset import AffinityDataModule

from commons.utils import load_yml_config, setup_neptune_logger
# from neptune_key import neptune_api_token

import argparse

import pandas as pd
import numpy as np
import os
import copy
import glob

def np_sigmoid(x):
    return 1 / (1+np.exp(x))

def onehot2cls(logits):
    # logits = torch.sigmoid(logits)
    # return (logits>.5).float().sum(-1)
    return np.sum((logits>.5), axis=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for neoantigen tasks.')
    parser.add_argument('--model', help='Model to use.', required=True)
    parser.add_argument('--gpu-id', type=int, help='GPU to run on 1~8.', required=True)
    parser.add_argument('--num-workers', type=int, help='Number of workers to use on dataloaders', default=8)
    parser.add_argument('--fold', type=int, help='Fold to use as validation')
    parser.add_argument('--resume_from_checkpoint', type=str,
                        help='Path/URL of the checkpoint from which training is resumed.', default=None)
    parser.add_argument('--data_filename', type=str, 
                        help='Path/URL of the training data', default=None)
    parser.add_argument('--test_data_filename', type=str,
                        help='Path/URL of the validation dataset.', default=None)    
    parser.add_argument('--test_params_rootdir', type=str,
                        help='Path/URL of the validation dataset.', default=None)
    parser.add_argument('--logger', type=bool, help='Whether to use neptune-logger', default=False)
    parser.add_argument('--seed', type=int, help='Seed to run model deterministally.', default=41)
    parser.add_argument('--classes', nargs='+',
                        help='List of class names in order', default=[])
    
    parser.add_argument('--pre_ln', type=bool,
                        help='which checkpoint to load', default=True)
    parser.add_argument('--init_type', type=str,
                        help='which checkpoint to load', default=None)
                        # help='which checkpoint to load', default='adaptive-profiling')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_layers_peptide', type=int, default=None)
    parser.add_argument('--n_layers_decoder', type=int, default=None)
    
    parser.add_argument('--mcdo_phase', type=int, default=1)
    args = parser.parse_args()
    

    # load configs
    config, _, _ = load_yml_config(f'configs/{args.model}/config_affinity.yml')
    config['seed'] = args.seed
    model_name = config['model_name']
    config['dataset_params']['seed'] = args.seed
    config['dataset_params']['fold'] = args.fold
    config['dataset_params']['num_workers'] = args.num_workers
    config['dataset_params']['data_filename'] = args.data_filename
    config['dataset_params']['test_data_filename'] = args.test_data_filename
    # 
    # config['dataset_params']['overwrite'] = True
    
    if args.classes != []:
        config['dataset_params']['classes'] = args.classes
    else:
        config['dataset_params']['classes'] = None
    
    config['train_params']['seed'] = args.seed
    if args.resume_from_checkpoint is not None:
        config['model_params']['pretrained_weights_path'] = args.resume_from_checkpoint
    
    if args.model == 'transformer':
        config['model_params']['pre_ln'] = args.pre_ln
        config['model_params']['init_type'] = args.init_type
    if args.n_layers_decoder is not None:
        config['model_params']['n_layers_decoder'] = args.n_layers_decoder
    
    # config['dataset_params']['inference'] = True
    config['dataset_params']['inference'] = False # For validation settings

    # # setup logger
    # logger = setup_neptune_logger(config, neptune_api_token) if args.logger else None
    
    # Import lightning module, dataset for chosen model
    if args.model == 'bertmhc':
        # from lightning_modules_bertmhc import AffinityPL
        from dataset_bertmhc import AffinityDataModule
    elif args.model == 'transphla':
        # from lightning_modules_transphla import AffinityPL
        from dataset_transphla import AffinityDataModule

    # set seeds
    pl.seed_everything(args.seed)

    # data processing
    if args.test_data_filename is not None:
        config['dataset_params']['test_data_filename'] = args.test_data_filename
    data = AffinityDataModule(config['dataset_params'])
    data.setup()

    if args.test_params_rootdir[-5:] == '.ckpt':
        checkpoints = [args.test_params_rootdir]
    else:
        rootdir = args.test_params_rootdir
        
        checkpoints = []
        for root, subdirs, filenames in os.walk(rootdir):
            for filename in filenames:
                if '.ckpt' in filename and 'last' not in filename:
                    checkpoints.append(root+'/'+filename)
        
    print(checkpoints)
    # checkdir = './gru/AF1-106/checkpoints/'
    # checkpoints = [checkdir+x for x in os.listdir(checkdir) if 'last.' not in x]
    # print(checkpoints)
    
    logits_list = []
    vars_list = []
    for checkpoint in checkpoints:
        
        config['model_params']['inference_weights_path'] = checkpoint
        
        # create torch lightning module
        kwarg_params = {'model_params', 'train_params', 'dataset_params', 'model_name'}
        
        pl_model = AffinityPL(**{k: config[k] for k in config.keys() & kwarg_params})

        # pl_model.checkpoint_dir = checkpoint
        
        trainer = pl.Trainer(gpus=[args.gpu_id])
        
        test_dataloader = data.test_dataloader()
        
        pl_model.test_dataloader = data.test_dataloader
        
        outputs = trainer.predict(
                            model=pl_model,
                            dataloaders=test_dataloader,
                            return_predictions=True)
        output_dict = {}
        for k in outputs[0].keys():
            if type(outputs[0][k]) == torch.Tensor:
                if outputs[0][k].is_cuda:
                    output_dict[k] = np.concatenate([x[k].detach().cpu().numpy() for x in outputs], axis=0)
                else:
                    output_dict[k] = np.concatenate([x[k] for x in outputs], axis=0)
            else:
                output_dict[k] = np.concatenate([x[k] for x in outputs], axis=0)
        
        # df = data.test_split.og_dataset
        # df = data.test_split.og_dataset.loc[data.test_split.dataset.index]
    
        # # df = pd.DataFrame()
        # # df['reg_preds'] = output_dict['reg_preds']
        # # df['reg_targets'] = output_dict['reg_targets']
        
        
        # df['hla_name'] = output_dict['hla_name']
        # df['peptide_name'] = output_dict['peptide_name']
        
        # df['mhc_type'] = output_dict['mhc_type']
        # df['method'] = output_dict['method']
        # df['technique'] = output_dict['technique']
        
        # if len(checkpoints) > 1:    
        #     # logits_list.append(df.loc[:,'reg_preds'].values.tolist())
        #     # vars_list.append(df.loc[:,'reg_preds_var'].values.tolist())
        logits_list.append(output_dict['reg_preds'])
        
        
    if len(checkpoints) > 1:
        # ensemble 'reg_pred' of outputs
        
        # Strategy 1: average all outputs
        
        logits_mean = np.mean(logits_list, axis=0)
        vars_mean = np.mean(vars_list, axis=0)
        
        # # Strategy 2: average except max, min values
        
        # logits_mean = (np.sum(logits_list, axis=0)-np.max(logits_list, axis=0)-np.min(logits_list, axis=0))\
        #                 /(len(logits_list)-2)
        # # vars_mean = (np.sum(vars_list, axis=0)-np.max(vars_list, axis=0)-np.min(vars_list, axis=0))\
        # #                 /(len(vars_list)-2)
        
    else:
        # df.to_csv()
        logits_mean = logits_list[0]
        
    
    # df = data.test_split.og_dataset
    # df = data.test_split.og_dataset.loc[data.test_split.dataset.index.tolist()]
    df = data.test_split.og_dataset.reset_index(drop=True).loc[data.test_split.dataset.index.tolist()]

    df['reg_preds'] = logits_mean
    # df = pd.DataFrame()
    # df['reg_preds'] = output_dict['reg_preds']
    # df['reg_targets'] = output_dict['reg_targets']
    # df['reg_preds_var'] = output_dict['reg_preds_var']
    
    # df['hla_name'] = output_dict['hla_name']
    # df['peptide_name'] = output_dict['peptide_name']
    
    # df['mhc_type'] = output_dict['mhc_type']
    # df['method'] = output_dict['method']
    # df['technique'] = output_dict['technique']
        
    # # df['normalized_pred'] = logits_mean
    
    # df['predictive_error'] = abs(logits_mean - df['normalized_label'])
        
    """
    evaluation
    """
    
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score,\
                                mean_absolute_error
    
    target = output_dict['reg_targets']
    perfs = {}
    
    # Only sort out preds and targets with inequality '='
    
    if args.classes != []:
        reg_preds_cls = onehot2cls(logits_mean)
        reg_targets_cls = onehot2cls(target)
        df['reg_preds_cls'] = reg_preds_cls
        df['reg_targets_cls'] = reg_targets_cls
        for i in range(len(args.classes)-1):
            df['reg_pred'+str(i)] = logits_mean[:,i]
    else:
        df['reg_preds'] = logits_mean
        df['reg_targets'] = target
        
    logits_mean = df.loc[df['assay_measurement_inequality']=='='].reset_index(drop=True)\
                    .loc[:,'reg_preds'].values
    target = df.loc[df['assay_measurement_inequality']=='='].reset_index(drop=True)\
                    .loc[:,'reg_targets'].values
    
    if len(args.classes) > 1:
        """
        Multi class metric -> averaged score of each class
        """
    
        # perfs['accuracy'] = np.mean([accuracy_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
        # perfs['precision'] = np.mean([precision_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
        # perfs['recall'] = np.mean([recall_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
        # perfs['f1'] = np.mean([f1_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
    
        perfs['auroc'] = np.mean([roc_auc_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
        perfs['auprc'] = np.mean([average_precision_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
    else:
        perfs['spearmanr'] = spearmanr(logits_mean, target)[0]
        perfs['pearsonr'] = pearsonr(logits_mean, target)[0]
        perfs['mae'] = mean_absolute_error(target, logits_mean)
        # perfs['mape'] = mean_absolute_percentage_error(target, logits_mean)
        
        # sort by top 100 predictions, get corr with target label
        logits_mean_top100 = np.sort(logits_mean)[::-1][:100]
        target_top100 = target[np.argsort(logits_mean)[::-1][:100]]
        
        # # perfs['top100 spearmanr'] = spearmanr(logits_mean_top100, target_top100)[0]
        # # perfs['top100 pearsonr'] = pearsonr(logits_mean_top100, target_top100)[0]
        # perfs['top10% IC50'] = df.sort_values('reg_preds', ascending=False).iloc[:len(df)//10]\
        #                             ['assay_quantitative_measurement'].mean()
        # perfs['top50 IC50'] = df.sort_values('reg_preds', ascending=False).iloc[:len(df)//2]\
        #                             ['assay_quantitative_measurement'].mean()
        
    # _, _, perfs['ece'], _ = calibration_guo(target, logits_mean, bins=10)
    
    # for k, v in perfs.items():
    #     print(k, v)
    print('\t'.join([str(v) for k,v in perfs.items()]))
    
    # if args.classes != []:
    #     vars_mean = np.mean(vars_mean, axis=-1)
    # df['reg_preds_var'] = vars_mean
        
    out_path = './data/inferred_csvs/'+'_'.join(args.test_params_rootdir.split('/')[1:2])+'.csv'
    df.to_csv(out_path)
    
    # """
    # sort by MCDO variance estimate, remove top 10k instances
    # """
    # if len(args.classes) > 1:
    #     vars_mean = np.mean(vars_mean, axis=1)
    # if 'regress' in config['dataset_params']['test_data_filename']:
    #     cutoff = 10000
    # elif 'class' in config['dataset_params']['test_data_filename']:
    #     cutoff = 5000
    
    # # df = pd.read_csv('./data/'+config['dataset_params']['dataset_name'] + '/' + config['dataset_params']['test_data_filename'], index_col=0)
    # df = data.test_split.dataset
    # df['MCDO_variance'] = vars_mean
    # df_drop_idxs = df.sort_values('MCDO_variance', ascending=False)[:cutoff].index
    # df = df.drop(df_drop_idxs).reset_index()\
    #             .to_csv('./data/'+config['dataset_params']['dataset_name'] + '/' + \
    #                 config['dataset_params']['test_data_filename'][:-4]+'_mcdo'+str(args.mcdo_phase)+'.csv')
    # # df = df.drop(df_drop_idxs).reset_index()\
    # #             .to_csv('./data/'+config['dataset_params']['dataset_name'] + '/' + \
    # #                 'mcdo_tmp.csv')

outputs = trainer.predict(
                    model=pl_model,
                    dataloaders=test_dataloader,
                    return_predictions=True)
output_dict = {}
for k in outputs[0].keys():
    try:
        if type(outputs[0][k]) == torch.Tensor:
            if outputs[0][k].is_cuda:
                output_dict[k] = np.concatenate([x[k].detach().cpu().numpy() for x in outputs], axis=0)
            else:
                output_dict[k] = np.concatenate([x[k] for x in outputs], axis=0)
        else:
            output_dict[k] = np.concatenate([x[k] for x in outputs], axis=0)
    except:
        output_dict[k] = [x[k] for x in outputs]

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
reducer = umap.UMAP(random_state=42)

# tensorboard/drocc_lr-2_r7/version_0/checkpoints/last.ckpt

# scaled_embed = StandardScaler().fit_transform(output_dict['embed'])
total_embeds = np.concatenate([output_dict['embed'],output_dict['adv_embed']], axis=0)
total_sources = output_dict['source'] + ['adversarial']*len(output_dict)
source2color = {'iedb':0, 'mhcflurry':1, 'ba_negative':2, 'adversarial': 3}
scaled_embed = StandardScaler().fit_transform(
                            )
embed = reducer.fit_transform(scaled_embed)
f, ax = plt.subplots(figsize=(17,15))
# points = ax.scatter(embed[:, 0], embed[:, 1], c=output_dict['reg_targets'], s=10, cmap="plasma")
points = ax.scatter(embed[:, 0], embed[:, 1], c=[source2color[x] for x in total_sources]
                    , s=20, cmap="plasma")
f.colorbar(points)
# plt.show()
plt.savefig('umap1.png', dpi=900)