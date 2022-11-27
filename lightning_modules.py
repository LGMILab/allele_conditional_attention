import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import model_checkpoint, early_stopping, \
                            LearningRateMonitor, StochasticWeightAveraging
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score,\
                        confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import mse_loss, relu, binary_cross_entropy_with_logits, binary_cross_entropy
import torch.nn.functional as F

from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from models.transformer import AffinityTransformerPretrainedSource, AffinityCrossTransformerPretrainedSource
from models.bertlike import AffinityBertlikePretrainedSource
from models.gru import AffinityGRU
from models.cnn import AffinityCNN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from schedulers import TriStageLRScheduler, InverseSQRTLRScheduler
import os
import copy
import yaml
import numpy as np
import math

from collections import OrderedDict


def bce_loss(preds: torch.Tensor, targets: torch.Tensor):
    return binary_cross_entropy_with_logits(preds, targets)

def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
    """
    Simple implementation of weighted MSE loss.
    :param preds: predictions tensor. [torch.Tensor]
    :param targets: targets tensor. [torch.Tensor]
    :param weights: sample loss weight [torch.Float]
    :return: mean weighted MSE loss. [torch.Float]
    """
    
    return (weights * (preds - targets) ** 2).mean()

def weighted_bce_loss(preds: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor):
    return binary_cross_entropy_with_logits(preds, targets, weight=weight)

def inequality_mse_loss(preds: torch.Tensor, targets: torch.Tensor, ineq: torch.Tensor, reduction: bool = True):
    """
    Implementation of MSE loss with inequalities. Assigns zeros loss if inequality is satisfied, else just calculates MSE.
    :param preds: predictions tensor. [torch.Tensor]
    :param targets: targets tensor. [torch.Tensor]
    :param ineq: wnequality measure as signed integer, {'>': -1, '=': 0, '<': 1}. [torch.Tensor]
    :param reduction: wheter calculate the mean of sample losses. [bool]
    :return: Mean or sample weighted MSE loss. [torch.Float / torch.Tensor]

    """
    loss = (1 - torch.abs(ineq)) * mse_loss(preds, targets, reduction='none') + relu(ineq * (targets - preds)) ** 2
    return torch.mean(loss) if reduction else loss

def weighted_inequality_mse_loss(preds: torch.Tensor, targets: torch.Tensor, 
                                ineq: torch.Tensor, weights: torch.Tensor, reduction: bool = True):
    """
    Implementation of MSE loss with inequalities. Assigns zeros loss if inequality is satisfied, else just calculates MSE.
    :param preds: predictions tensor. [torch.Tensor]
    :param targets: targets tensor. [torch.Tensor]
    :param ineq: wnequality measure as signed integer, {'>': -1, '=': 0, '<': 1}. [torch.Tensor]
    :param reduction: wheter calculate the mean of sample losses. [bool]
    :return: Mean or sample weighted MSE loss. [torch.Float / torch.Tensor]

    """
    loss = (1 - torch.abs(ineq)) * mse_loss(preds, targets, reduction='none') + relu(ineq * (targets - preds)) ** 2
    return torch.mean(weights * loss) if reduction else weights * loss
    
def np_sigmoid(x):
    return 1 / (1+np.exp(x))

def onehot2cls(logits: torch.tensor):
    return (logits>.5).float().sum(-1)

class AffinityPL(pl.LightningModule):
    def __init__(self, model_params: dict, train_params: dict, dataset_params: dict, model_name: str):
        """
        Main pytorch lightning module for binding affinity task. Configuration files are read from "./configs'
        :param model_params: Model parameters configuration. [dict]
        :param train_params:  Training parameters configuration. [dict]
        :param dataset_params: Dataset parameters configuration. [dict]
        :param model_name: Name of the current model. [str]
        """
        super().__init__()

        # module parameters
        self.model_params = model_params
        self.train_params = train_params
        self.learning_rate = self.train_params['optimizer']['lr']
        self.weight_decay = self.train_params['optimizer']['wd']
        self.use_assay_features = dataset_params['use_assay_features']
        self.peptide_max_len = dataset_params['peptide_max_len']
        self.hla_max_len = dataset_params['hla_max_len']
        self.results_filename = None
        self.model_name = model_name

        self.classes = None
        num_classes = 1
            
        if model_name == 'transformer':
            self.model = AffinityTransformerPretrainedSource(token_dim_peptide=self.model_params['token_dim_peptide'],
                                                             token_dim_hla=self.model_params['token_dim_hla'],
                                                             hidden_dim=self.model_params['hidden_dim'],
                                                             n_heads=self.model_params['n_heads'],
                                                             n_layers_decoder=self.model_params['n_layers_decoder'],
                                                             peptide_max_len=self.peptide_max_len,
                                                             hla_max_len=self.hla_max_len,
                                                             cnn_pool_channels=self.model_params['cnn_pool_channels'],
                                                             dropout=self.model_params['dropout'],
                                                             activation=self.model_params['activation'],
                                                             num_classes=num_classes,
                                                             emb_type = dataset_params['emb_type'],
                                                             pool_type=self.model_params['pool_type'])

        if model_name == 'cross_transformer':
            self.model = AffinityCrossTransformerPretrainedSource(token_dim_peptide=self.model_params['token_dim_peptide'],
                                                             token_dim_hla=self.model_params['token_dim_hla'],
                                                             hidden_dim=self.model_params['hidden_dim'],
                                                             n_heads=self.model_params['n_heads'],
                                                             n_layers_decoder=self.model_params['n_layers_decoder'],
                                                             peptide_max_len=self.peptide_max_len,
                                                             hla_max_len=self.hla_max_len,
                                                             cnn_pool_channels=self.model_params['cnn_pool_channels'],
                                                             dropout=self.model_params['dropout'],
                                                             activation=self.model_params['activation'],
                                                            #  use_assay_features=self.use_assay_features,
                                                            #  assay_features_dim=dataset_params['assay_features_dim'],
                                                             num_classes=num_classes,
                                                             emb_type = dataset_params['emb_type'],
                                                             pool_type=self.model_params['pool_type'])
            
        if model_name == 'bertlike':
            self.model = AffinityBertlikePretrainedSource(token_dim_peptide=self.model_params['token_dim_peptide'],
                                                             token_dim_hla=self.model_params['token_dim_hla'],
                                                             hidden_dim=self.model_params['hidden_dim'],
                                                             n_heads=self.model_params['n_heads'],
                                                             n_layers_decoder=self.model_params['n_layers_decoder'],
                                                             peptide_max_len=self.peptide_max_len,
                                                             hla_max_len=self.hla_max_len,
                                                             cnn_pool_channels=self.model_params['cnn_pool_channels'],
                                                             dropout=self.model_params['dropout'],
                                                             activation=self.model_params['activation'],
                                                             num_classes=num_classes,
                                                             emb_type = dataset_params['emb_type'],
                                                             pool_type=self.model_params['pool_type'])

        if model_name == 'gru':
            self.model = AffinityGRU(token_dim_peptide=self.model_params['token_dim_peptide'],
                                     token_dim_hla=self.model_params['token_dim_hla'],
                                     hidden_dim_peptide=self.model_params['hidden_dim_peptide'],
                                     hidden_dim_hla=self.model_params['hidden_dim_hla'],
                                     seq_len_hla=self.model_params['seq_len_hla'],
                                     n_layers_peptide=self.model_params['n_layers_peptide'],
                                     cnn_out_channels_hla=self.model_params['cnn_out_channels_hla'],
                                     dropout=self.model_params['dropout'],
                                     num_classes=num_classes,
                                     emb_type = dataset_params['emb_type'])
        
        if model_name == 'cnn':
            self.model = AffinityCNN(token_dim_peptide=self.model_params['token_dim_peptide'],
                                     token_dim_hla=self.model_params['token_dim_hla'],
                                     hidden_dim_peptide=self.model_params['hidden_dim_peptide'],
                                     hidden_dim_hla=self.model_params['hidden_dim_hla'],
                                     seq_len_hla=self.model_params['seq_len_hla'],
                                     n_layers_peptide=self.model_params['n_layers_peptide'],
                                     cnn_out_channels_hla=self.model_params['cnn_out_channels_hla'],
                                     dropout=self.model_params['dropout'],
                                     num_classes=num_classes,
                                     emb_type = dataset_params['emb_type'])
        
        # load layers from pretrained model and freeze everything except last fully connected ones.
        if model_params['pretrained']:
            
            # ignore linear layers and load weights
            state_dict = torch.load(open(model_params['pretrained_weights_path'], 'rb'), 
                                    map_location=torch.device('cpu'))['state_dict']
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.linear_block')}
            self.load_state_dict(state_dict, strict=False)

            # freeze layers
            for name, layer in self.model.named_children():
                if not name.startswith('linear_b'):
                    layer.requires_grad_(False)
            
        # for inference usage
        
        if self.model_params['inference_weights_path'] is not None:
            
            state_dict = torch.load(open(model_params['inference_weights_path'], 'rb'), 
                                    map_location=torch.device('cpu'))['state_dict']
            self.load_state_dict(state_dict, strict=False)

            # freeze layers
            for name, layer in self.model.named_children():
                layer.requires_grad_(False)                                                      

        """ 
        this should be turned True if embeddings are needed for each output.
        set this true after lightning module class initialized.
        """
        self.get_embeddings = False
        self.get_attention_weights = False
        self.logger_type = None
        self.assay_onehot_encoder = None

    def forward(self, batch: dict):
        """
        Set for inference phase.
        """
        
        # forward pass
        if self.get_embeddings: # Only implemented for Transformer, CrossTransformer
            if self.use_assay_features:
                    outputs, embed = self.model(batch['peptide'], batch['hla'], 
                                         batch['assay_features'].squeeze(),
                                         get_embeddings=self.get_embeddings)
            else:
                outputs, embed = self.model(batch['peptide'], batch['hla'],
                                        get_embeddings=self.get_embeddings)
                                
            outputs = outputs.squeeze().cpu().detach().numpy()
            embed = embed.squeeze().cpu().detach().numpy()
            
        elif self.get_attention_weights: # Only implemented for Transformer, CrossTransformer

            outputs, embed = self.model(batch['peptide'], batch['hla'],
                                    get_attention_weights=self.get_attention_weights)
            
            outputs = outputs.squeeze().cpu().detach().numpy()
            attn_weights = {}
            for k, v in embed.items():
                attn_weights[k] = v.squeeze().cpu().detach().numpy()
        
        else:

            outputs = self.model(batch['peptide'], batch['hla']).squeeze()
                                
        output_dict = {'reg_preds': outputs,
                'hla_name': batch['hla_name'],
                'peptide_name': batch['peptide_name'],
                }


    
        if 'reg_target' in batch.keys():
            output_dict['reg_targets'] = batch['reg_target']

        if self.get_embeddings:
            output_dict['embed'] = embed
            
        if self.get_attention_weights:
            output_dict['attn_weights'] = attn_weights
            
        return output_dict

    def training_step(self, batch: dict, batch_idx: int):
        """
        Gets one batch from self.train_dataloader() and runs one training batch.
        :param batch: batch from dataloader. [dict]
        :param batch_idx: batch index. [int]
        """

        outputs = self.model(batch['peptide'], batch['hla']).squeeze()
            
        if self.train_params['use_inequality_loss']:
            # inequality mse loss
            loss_reg = inequality_mse_loss(outputs, batch['reg_target'], batch['inequality']).sum()
        else:
            loss_reg = mse_loss(outputs, batch['reg_target'])

        losses_dict = {'Weighted MSE loss': loss_reg}

        if len(losses_dict) > 1:
            # add other losses to progress bar since total loss is added automatically
            self.log_dict(losses_dict, prog_bar=True, logger=False)

        # formatting for tensorboard logging
        losses_dict = {f'loss/{k}': v for k, v in losses_dict.items()}

        # training_step must always return None, a Tensor, or a dict with at least one key being 'loss'
        losses_dict['loss'] = loss_reg.sum()
        return {'reg_preds': outputs,
                'reg_targets': batch['reg_target'],
                'loss': loss_reg,
                'hla_name': batch['hla_name'],
                'peptide_name': batch['peptide_name'],
                'ineqs': batch['inequality'],
                'ineqs_symbol': batch['inequality_symbol'],
                }

    def training_step_end(self, outputs):
        outputs['loss'] = outputs['loss'].sum()
        return outputs

    def training_epoch_end(self, outputs):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = torch.hstack([o['loss'] for o in outputs]).mean()

        self.log('train_loss', loss)

        # if logger is setup, log metrics
        if self.logger:
            
            hla = [i for o in outputs for i in o['hla_name']]
            peptide = [i for o in outputs for i in o['peptide_name']]
            ineq_symbols = [i for o in outputs for i in o['ineqs_symbol']]
            ineqs = torch.hstack([i for o in outputs for i in o['ineqs']]).cpu().detach()
            
            reg_preds = torch.hstack([i for o in outputs for i in o['reg_preds']]).cpu().detach()
            reg_targets = torch.hstack([i for o in outputs for i in o['reg_targets']]).cpu().detach()

            # predictions
            res = pd.DataFrame({'hla': hla,
                                'peptide': peptide,
                                'reg_target': reg_targets,
                                'inequality': ineq_symbols,
                                'reg_pred': reg_preds,

                                })
            
            # spearman score
            spearman_data = res[res.inequality == '=']
            
            spearman_corr = spearmanr(spearman_data.reg_pred, spearman_data.reg_target)
            
            if self.logger_type == 'neptune':
                self.logger.log_metric('train_spearman_corr', spearman_corr[0])
            elif self.logger_type == 'tensorboard':
                self.log('train_spearman_corr', spearman_corr[0])
                

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Gets one batch from self.test_dataloader() and runs one validation batch.
        :param batch: batch from dataloader. [dict]
        :param batch_idx: batch index. [int]
        """
        
        outputs = self.model(batch['peptide'], batch['hla']).squeeze()

        if self.train_params['use_inequality_loss']:
            # inequality mse loss
            loss_reg = inequality_mse_loss(outputs, batch['reg_target'], batch['inequality']).sum()
        else:
            loss_reg = mse_loss(outputs, batch['reg_target'])

        return {'reg_preds': outputs,
                'reg_targets': batch['reg_target'],
                'loss': loss_reg,
                'hla_name': batch['hla_name'],
                'peptide_name': batch['peptide_name'],
                'ineqs': batch['inequality'],
                'ineqs_symbol': batch['inequality_symbol'],
                }

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        """
        Calculate loss and metrics after one validation epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = torch.hstack([o['loss'] for o in outputs]).mean()

        self.log('val_loss', loss, sync_dist=True) # multi gpu usage
        
        # if logger is setup, log metrics
        if self.logger:
            hla = [i for o in outputs for i in o['hla_name']]
            peptide = [i for o in outputs for i in o['peptide_name']]

            ineqs = torch.hstack([i for o in outputs for i in o['ineqs']]).cpu().detach()
            ineq_symbols = [i for o in outputs for i in o['ineqs_symbol']]
        
            reg_preds = torch.hstack([i for o in outputs for i in o['reg_preds']]).cpu()
            reg_targets = torch.hstack([i for o in outputs for i in o['reg_targets']]).cpu()
            
            res = pd.DataFrame({'hla': hla,
                                'peptide': peptide,
                                'reg_target': reg_targets,
                                'inequality': ineq_symbols,
                                'reg_pred': reg_preds,
                                })
            

            # spearman score
            spearman_data = res[res.inequality == '=']
            spearman_corr = spearmanr(spearman_data.reg_pred, spearman_data.reg_target)
            if self.logger_type == 'neptune':
                self.logger.log_metric('valid_spearman_corr', spearman_corr[0])
            elif self.logger_type == 'tensorboard':
                self.log('valid_spearman_corr', spearman_corr[0])

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Scheduler configuration can be changed in the config files (config/[model_name]), over train_params.
        """

        # optimizers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # schedulers
        # TODO: add schuler parameters restrictions to only implemented schdulers.
        scheduler_name = self.train_params['scheduler']['name']
        if scheduler_name == 'TriStageLRScheduler':
            scheduler = TriStageLRScheduler(optimizer, **self.train_params['scheduler']['params'])
            return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        
        elif scheduler_name == 'InverseSQRTLRScheduler':
            scheduler = InverseSQRTLRScheduler(optimizer, **self.train_params['scheduler']['params'])
            return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        
        elif scheduler_name == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif scheduler_name == 'CosineWithWarmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif scheduler_name == 'CosineWithHardWarmup':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 
                    "monitor": self.train_params['monitor']}

        else:
            return optimizer

    def configure_callbacks(self):
        """
        Configure callbacks for torch lightning module. Refere to https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html.
        """
        checkpoint = model_checkpoint.ModelCheckpoint(monitor=self.train_params['monitor'],
                                                      verbose=True,
                                                      mode=self.train_params['monitor_mode'],
                                                      save_top_k=self.train_params['save_top_k'],
                                                      save_last=True)
        earlystop = early_stopping.EarlyStopping(monitor=self.train_params['monitor'],
                                                 verbose=True,
                                                 mode=self.train_params['monitor_mode'],
                                                 patience=self.train_params['patience'])

        callbacks = [checkpoint]
        if self.train_params['early_stop']:
            callbacks.append(earlystop)
        if self.logger:
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor)

        return callbacks
