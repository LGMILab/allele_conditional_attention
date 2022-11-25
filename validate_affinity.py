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
import math
import copy
import glob

def np_sigmoid(x):
    return 1 / (1+np.exp(x))

def onehot2cls(logits):
    return np.sum((logits>.5), axis=-1)

def normalize(x):
    return 1-math.log(min(max(x,1.),50000.))/math.log(50000)

def reverse_normalize(y):
    '''
    y = 1-log(x)/log(50000)
    x = exp(log(50000)*(1-y))
    '''
    return min(math.exp(math.log(50000)*(1-y)), 50000.)

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
    
    parser.add_argument('--batch_size', type=int, default=512)
    
    parser.add_argument('--mcdo_phase', type=int, default=1)

    parser.add_argument('--emb_type', type=str,
                        choices={"aa2", "aa+esm", "re"},
                        default="aa+esm")
    parser.add_argument('--pool_type', type=str,
                        choices={"average", "conv", "token"},
                        default="conv")
                        # default="average")

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
    
    # if args.classes != []:
    #     config['dataset_params']['classes'] = args.classes
    # else:
    #     config['dataset_params']['classes'] = None
    
    config['train_params']['seed'] = args.seed
    if args.resume_from_checkpoint is not None:
        config['model_params']['pretrained_weights_path'] = args.resume_from_checkpoint
    
    # config['dataset_params']['inference'] = True
    config['dataset_params']['inference'] = False # For validation settings

    config['dataset_params']['emb_type'] = args.emb_type
    if config['dataset_params']['emb_type'] == 'esm2':
        # config['dataset_params']['use_esm'] = True
        config['model_params']['token_dim_peptide'] = config['model_params']['token_dim_hla']
        config['dataset_params']['peptide_max_len'] += 1
    if config['dataset_params']['emb_type'] in ['re']:
        config['dataset_params']['hla_max_len'] -= 1

    config['model_params']['pool_type'] = args.pool_type
    # add one more layer for aggregating attentions w.r.t. tokens
    if config['model_params']['pool_type'] == 'token':
        config['model_params']['n_layers_decoder'] += 1
    
    # Import lightning module, dataset for chosen model
    if args.model == 'bertmhc':
        from dataset_bertmhc import AffinityDataModule
    elif args.model == 'transphla':
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
    
    logits_list = []
    vars_list = []
    for checkpoint in checkpoints:
        
        config['model_params']['inference_weights_path'] = checkpoint
        
        # create torch lightning module
        kwarg_params = {'model_params', 'train_params', 'dataset_params', 'model_name'}
        
        pl_model = AffinityPL(**{k: config[k] for k in config.keys() & kwarg_params})
        
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
        
        logits_list.append(output_dict['reg_preds'])
        
    if len(checkpoints) > 1:
        # ensemble 'reg_pred' of outputs
        
        # Strategy 1: average all outputs
        
        logits_mean = np.mean(logits_list, axis=0)
        vars_mean = np.mean(vars_list, axis=0)
        
    else:
        # df.to_csv()
        logits_mean = logits_list[0]
        
    df = data.test_split.og_dataset.reset_index(drop=True).loc[data.test_split.dataset.index.tolist()]
    df['reg_preds'] = logits_mean
        
    """
    evaluation
    """
    
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score,\
                                mean_absolute_error
    
    target = output_dict['reg_targets']
    perfs = {}
    
    # if args.classes != []:
    #     reg_preds_cls = onehot2cls(logits_mean)
    #     reg_targets_cls = onehot2cls(target)
    #     df['reg_preds_cls'] = reg_preds_cls
    #     df['reg_targets_cls'] = reg_targets_cls
    #     for i in range(len(args.classes)-1):
    #         df['reg_pred'+str(i)] = logits_mean[:,i]
    # else:
    df['reg_preds'] = logits_mean
    df['reg_targets'] = target

    # Only sort out preds and targets with inequality '='
    logits_mean = df.loc[df['assay_measurement_inequality']=='='].reset_index(drop=True)\
                    .loc[:,'reg_preds'].values
    target = df.loc[df['assay_measurement_inequality']=='='].reset_index(drop=True)\
                    .loc[:,'reg_targets'].values
    
    # if len(args.classes) > 1:
    #     """
    #     Multi class metric -> averaged score of each class
    #     """

    #     perfs['auroc'] = np.mean([roc_auc_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
    #     perfs['auprc'] = np.mean([average_precision_score(target[:,i], logits_mean[:,i]) for i in range(len(args.classes)-1)])
    # else:

    perfs['spearmanr'] = spearmanr(logits_mean, target)[0]
    perfs['pearsonr'] = pearsonr(logits_mean, target)[0]
    perfs['mae'] = mean_absolute_error(target, logits_mean)

    # classification
    threshold = normalize(500.)
    target_cls = [1. if x > threshold else 0. for x in target]
    logits_mean_cls = np.clip(logits_mean, a_min=0., a_max=1.)

    perfs['auroc'] = roc_auc_score(target_cls, logits_mean_cls)
    perfs['aupr'] = average_precision_score(target_cls, logits_mean_cls)

    # top 100  -> similar to precision K, showing relevance of queried top K items
    perfs['top 100 mean BA'] = np.mean(target[np.argsort(logits_mean)[::-1][:100]])
    perfs['top 50 mean BA'] = np.mean(target[np.argsort(logits_mean)[::-1][:50]])
    
    print('\t'.join([str(k) for k,v in perfs.items()]))
    print('\t'.join([str(v) for k,v in perfs.items()]))
    
    out_path = './data/inferred_csvs/'+'_'.join(args.test_params_rootdir.split('/')[1:2])+'.csv'
    df.to_csv(out_path)
    