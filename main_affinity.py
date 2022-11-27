import pytorch_lightning as pl

from lightning_modules import AffinityPL
from dataset import AffinityDataModule


from pytorch_lightning.plugins import DDPPlugin

from commons.utils import load_yml_config, setup_neptune_logger, setup_tensorboard_logger
import os
from pathlib import Path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for neoantigen tasks.')
    parser.add_argument('--mhc_class', type=int, default=1)
    parser.add_argument('--model', help='Model to use.', 
                        choices={"bertlike", "transformer", "cross_transformer", "gru", "cnn"}, 
                        required=True)
    parser.add_argument('--gpu-id', type=str, 
                        help='GPU to run on 1~8. To use multi-gpu settings, set args like 4,5,6',
                        required=True)
    parser.add_argument('--num-workers', type=int, help='Number of workers to use on dataloaders', default=8)
    parser.add_argument('--fold', type=int, help='Fold to use as validation', default=None)
    parser.add_argument('--resume-from-checkpoint', type=str,
                        help='Path/URL of the checkpoint from which training is resumed.', default=None)
    parser.add_argument('--logger', type=str, help='Wheter to use neptune-logger or tensorboard. {neptune,tensorboard} ', 
                        default='tensorboard')
    parser.add_argument('--seed', type=int, help='Seed to run model deterministally.', default=41)
    parser.add_argument('--default_root_dir', type=str, 
                        help='Path/URL of the checkpoint from which training is resumed.', default=None)
    parser.add_argument('--data_filename', type=str, 
                        help='Path/URL of the training data', default=None)
    parser.add_argument('--test_data_filename', type=str, 
                        help='Path/URL of the valid data', default=None)
    parser.add_argument('--classes', nargs='+',
                        help='List of class names in order', default=[])
    parser.add_argument('--pretrained_weights_path', type=str,
                        help='which checkpoint to load', default=None)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--emb_type', type=str,
                        choices={"aa2", "aa+esm", "re"},
                        default="aa+esm")

    parser.add_argument('--pool_type', type=str,
                        choices={"average", "conv", "token"},
                        default="conv")           

    args = parser.parse_args()

    # load configs
    if args.mhc_class == 1:
        config, _, _ = load_yml_config(f'configs/{args.model}/config_affinity.yml')
    elif args.mhc_class == 2:
        config, _, _ = load_yml_config(f'configs/{args.model}2/config_affinity.yml')
    config['seed'] = args.seed
    model_name = config['model_name']
    config['dataset_params']['seed'] = args.seed
    config['dataset_params']['fold'] = args.fold
    config['dataset_params']['num_workers'] = args.num_workers
    config['train_params']['seed'] = args.seed
    config['dataset_params']['batch_size'] = args.batch_size
    config['dataset_params']['inference'] = False

    if config['dataset_params']['classes'] is not None:
        config['model_params']['num_classes'] = len(config['dataset_params']['classes'])
    if args.data_filename is not None:
        config['dataset_params']['data_filename'] = args.data_filename
        config['dataset_params']['test_data_filename'] = args.test_data_filename

    if args.classes != []:
        config['dataset_params']['classes'] = args.classes
    else:
        config['dataset_params']['classes'] = None

    if args.pretrained_weights_path is not None:
        config['model_params']['pretrained'] = True
        config['model_params']['pretrained_weights_path'] = args.pretrained_weights_path
        
    if ',' not in args.gpu_id:
        args.gpu_id = [int(args.gpu_id)]

    config['dataset_params']['emb_type'] = args.emb_type

    config['model_params']['pool_type'] = args.pool_type
    # add one more layer for aggregating attentions w.r.t. tokens
    if config['model_params']['pool_type'] == 'token':
        config['model_params']['n_layers_decoder'] += 1
    
    # setup logger
    if args.logger == 'neptune':
        tags = []
        tags.append(args.default_root_dir.split('/')[-1])
        logger = setup_neptune_logger(config, neptune_api_token, tags) if args.logger else None
    elif args.logger == 'tensorboard':
        logger = setup_tensorboard_logger(config, args) if args.logger else None

        if args.model == 'bertmhc':
            from dataset_bertmhc import AffinityDataModule
    # set seeds
    pl.seed_everything(args.seed)

    # create torch lightning module
    
    kwarg_params = {'model_params', 'train_params', 'dataset_params', 'model_name'}
    pl_model = AffinityPL(**{k: config[k] for k in config.keys() & kwarg_params})
    pl_model.logger_type = args.logger

    # data processing
    data = AffinityDataModule(config['dataset_params'])
    data.setup()
    pl_model.assay_onehot_encoder = data.assay_onehot_encoder
    
    # set ckpt save path
    if args.default_root_dir is not None:
        if not os.path.exists(args.default_root_dir):
            path = Path(args.default_root_dir)
            path.mkdir(parents=True, exist_ok=True)
    
    trainer = pl.Trainer(
                        gpus=args.gpu_id,
                         accelerator='ddp',
                         precision=16,
                         check_val_every_n_epoch=1,
                         amp_backend='native',
                         max_epochs=1500,
                         checkpoint_callback=False,
                         weights_summary='full',
                         num_sanity_val_steps=0,
                         logger=logger,
                         default_root_dir=args.default_root_dir,
                         resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.fit(pl_model, data)
