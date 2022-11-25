import os
import pickle
from pathlib import Path
from typing import Optional

import esm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, ConcatDataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from commons.utils import print_color, get_blosum62_matrix
import pytorch_lightning as pl
from copy import deepcopy

from ems.generate_esm import get_esm_representations
from commons.augmentation import mhc_i_imm_aug

from tape.tokenizers import TAPETokenizer
from tape.datasets import pad_sequences as tape_pad

class AffinityDataModule(pl.LightningDataModule):

    def __init__(self, dataset_params):
        """
        Torch lightning datamodule for affinity dataset.
        :param dataset_params:
        """
        super().__init__()
        self.data_dir = dataset_params['data_dir']
        self.data_filename = dataset_params['data_filename']
        self.test_data_filename = dataset_params['test_data_filename']
        self.batch_size = dataset_params['batch_size']
        self.dataset_name = dataset_params['dataset_name']
        self.val_ratio = dataset_params['val_ratio']
        self.num_workers = dataset_params['num_workers']
        self.seed = dataset_params['seed']
        self.weighted_sampling = dataset_params['weighted_sampling']
        self.weighted_sampling_bins = dataset_params['weighted_sampling_bins']
        self.peptide_max_len = dataset_params['peptide_max_len']
        self.use_blosum = dataset_params['use_blosum']
        self.msa = dataset_params['msa']
        self.use_assay_features = dataset_params['use_assay_features']
        self.use_test_as_val = dataset_params['use_test_as_val']
        self.pca_components = dataset_params['pca_components']
        self.fold = dataset_params['fold']
        self.overwrite = dataset_params['overwrite']
        self.hla_sequences_filename = dataset_params['hla_sequences_filename']
        self.onehot_encoder = None
        self.classes = dataset_params['classes']
        self.inference = dataset_params['inference']
        

        # load datasets
        self.dataset = pd.read_csv(f'{self.data_dir}/data/affinity/{self.data_filename}')
        if self.test_data_filename is not None:
            self.test_dataset = pd.read_csv(f'{self.data_dir}/data/affinity/{self.test_data_filename}')

        self.train_split, self.val_split, self.test_split, self.inference_split, self.sampler = None, None, None, None, None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datamodules. Train and test data in different source files.
        Creates validation split with samples dcorresponding to "fold".
        Rest of data used for training.
        """
        if self.fold is not None:
            train_dataset = self.dataset[self.dataset.fold != self.fold]
            val_dataset = self.dataset[self.dataset.fold == self.fold]
        else:
            train_dataset = self.dataset
            val_dataset = self.test_dataset
            
            # train_dataset = self.dataset[self.dataset.fold != self.fold]
            # val_dataset = self.dataset[self.dataset.fold == self.fold]

        # if using assay features create one hot encoder for assay features (using train data)
        self.assay_onehot_encoder = OneHotEncoder(categories=[['purified MHC', 'cellular MHC'],
                                                              ['direct', 'competitive'],
                                                              ['fluorescence', 'radioactivity']]) if self.use_assay_features else None                                                              
        if self.use_assay_features:
            self.assay_onehot_encoder.fit(train_dataset[['assay_mhc_type', 'assay_method', 'assay_technique']].values.tolist())

        # train dataset
        self.train_split = AffinityDataset(dataset=train_dataset,
                                           data_filename=self.data_filename,
                                           root_folder=self.data_dir,
                                           peptide_max_len=self.peptide_max_len,
                                           use_blosum=self.use_blosum,
                                           msa=self.msa,
                                           use_assay_features=self.use_assay_features,
                                           assay_onehot_encoder=self.assay_onehot_encoder,
                                           pca_components=self.pca_components,
                                           hla_sequences_filename=self.hla_sequences_filename,
                                           split='train',
                                           fold=self.fold,
                                           overwrite=self.overwrite,
                                           classes=self.classes,
                                           inference=False)

        print('number of Train dataset instances : '+str(len(self.train_split)))

        # validation dataset
        self.val_split = AffinityDataset(dataset=val_dataset,
        # self.val_split = AffinityDataset(dataset=train_dataset,
                                         data_filename=self.data_filename,
                                         root_folder=self.data_dir,
                                         peptide_max_len=self.peptide_max_len,
                                         use_blosum=self.use_blosum,
                                         msa=self.msa,
                                         use_assay_features=self.use_assay_features,
                                         assay_onehot_encoder=self.assay_onehot_encoder,
                                         pca_components=self.pca_components,
                                         hla_sequences_filename=self.hla_sequences_filename,
                                         split='val',
                                         fold=self.fold,
                                         overwrite=self.overwrite,
                                         classes=self.classes,
                                         inference=False)
        print('number of Valid dataset instances : '+str(len(self.val_split)))
        
        print(self.train_split.dataset.head())
        
        print(self.val_split.dataset.head())

        # if there is no test data use validation split as test
        if self.test_data_filename is None:
            self.test_split = deepcopy(self.val_split)
        else:
            self.test_split = AffinityDataset(dataset=self.test_dataset,
                                                data_filename=self.test_data_filename,
                                                root_folder=self.data_dir,
                                                peptide_max_len=self.peptide_max_len,
                                                use_blosum=self.use_blosum,
                                                msa=self.msa,
                                                use_assay_features=self.use_assay_features,
                                                assay_onehot_encoder=self.assay_onehot_encoder,
                                                pca_components=self.pca_components,
                                                hla_sequences_filename=self.hla_sequences_filename,
                                                split='test',
                                                fold=0,
                                                overwrite=self.overwrite,
                                                classes=self.classes,
                                                inference=self.inference)
            print('number of Test dataset instances : '+str(len(self.test_split)))
      
        # balanced sampling (10 bins)
        if self.weighted_sampling:
            # TODO: create argument for binning process v
            # bin continous label
            
            # bins = 10
            # binned_samples = pd.cut(self.train_split.dataset.normalized_label, bins, labels=range(bins)).reset_index()['normalized_label']
            
            binned_samples = pd.cut(self.train_split.dataset.normalized_label, 
                                    self.weighted_sampling_bins, 
                                    labels=range(self.weighted_sampling_bins))\
                                .reset_index()['normalized_label']
            ratios = binned_samples.shape[0] / binned_samples.value_counts()  # 1/count_ratio
            samples_weight = [ratios[i] for i in binned_samples]
            samples_weight /= max(samples_weight)
            
            self.sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            self.sampler = None
        return

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True,
                        collate_fn=self.train_split.collate_fn,
                          num_workers=self.num_workers, pin_memory=True, sampler=None)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, shuffle=False, 
                        collate_fn=self.val_split.collate_fn,
                          pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False, 
                        collate_fn=self.test_split.collate_fn,
                          pin_memory=True, num_workers=self.num_workers)


class AffinityDataset(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame,
                 data_filename: str,
                 root_folder: str,
                 peptide_max_len: int,
                 use_blosum: bool,
                 msa: bool,
                 use_assay_features: bool,
                 assay_onehot_encoder: OneHotEncoder,
                 pca_components: int,
                 hla_sequences_filename: str,
                 split: str,
                 fold: int,
                 overwrite: bool,
                 classes: list,
                 inference: bool):
        
        """
        Torch dataset for binding affinity task. Generates features for each HLA and peptide sequence
        using AAindex (aminoacid level), BLOSUM62 (aminoacid level) matrix and assay features (sequence level).
        :param dataset: dataset as tabular date. [pd.DataFrame]
        :param data_filename: dataset filename. [str]
        :param root_folder: path of the root folder [str]
        :param peptide_max_len: longest peptide sequence in the dataset. [str]
        :param use_blosum: wheter concatenate BLOSUM62 matrix features to inputs. [bool]
        :param use_assay_features: wheter concatenate assay features to inputs. [bool]
        # TODO: assert if the encoder was already fitted
        :param assay_onehot_encoder: sklearn one hot encoder, already fitted with training data. [OneHotEncoder]
        :param pca_components: number of AAindex principal components to use. [int]
        :param hla_sequences_filename: hla-name to hla-sequence filename. [str]
        :param split: used for saving preprocessed files. [str]
        :param fold: used for saving preprocessed files. Assumes pre-generated folds [str]
        :param overwrite: wether to overwrite preprocessed files (if exist). [bool]
        :param classes: list of classes for classification task, default set to None. Shall be listed in ascending order. [int]
        """

        self.root_folder = root_folder
        self.use_assay_features = use_assay_features
        self.assay_onehot_encoder = assay_onehot_encoder
        self.classes = classes
        self.inference = inference
        
        # deepcopy to avoid problems with dataframe in torch lightning data module
        self.dataset = dataset.copy(deep=True)
        

        # aminoacid features and normalization
        self.aminoacid_features = pd.read_csv(f'{root_folder}/data/aminoacid_features.csv', index_col='aminoacid')

        # pca projection for aminoacid features (AAIndex)
        self.pca = PCA(pca_components)
        self.aminoacid_features = pd.DataFrame(self.pca.fit_transform(self.aminoacid_features.values),
                                               columns=[f'pca_{i}' for i in range(pca_components)],
                                               index=self.aminoacid_features.index)

        # get blossum matrix and attach to aminoacid faetures
        if use_blosum:
            # TODO: test thing part for bugs, merge might introduce some undesired behaviour
            blosum_features = get_blosum62_matrix()
            
            # take only columns of blosum matrix present in the index of the aminoacid_features dataframe
            
            # self.aminoacid_features = pd.merge(blosum_features[self.aminoacid_features.index],
            self.aminoacid_features = pd.merge(blosum_features[self.aminoacid_features.index].astype(float),
                                               self.aminoacid_features,
                                               left_index=True,
                                               right_index=True)

        # normalize features
        self.amino_scaler = StandardScaler()
        self.aminoacid_features.loc[:, :] = self.amino_scaler.fit_transform(self.aminoacid_features)

        # ignore samples with aminoacids outside of 20 essential
        allowed_vocabulary = set(self.aminoacid_features.index)
        self.dataset.loc[:, 'peptide'] = self.dataset.peptide.str.upper()
        self.dataset.peptide = self.dataset.peptide.str.replace('-', '')
        self.dataset = self.dataset[self.dataset.peptide.apply(lambda x: set(x).issubset(allowed_vocabulary))]
        
        # groupby and take median for cases with multiple measuerements
        # gt and lt also get grouped separately to avoid problems between inequality types
        
        # Drop rows where labels are not in given possible classes
        
        if classes is not None:
            self.dataset = self.dataset.loc[self.dataset['assay_qualitative_measure'].isin(classes)]
        
        # Store dataset before reset_index()
        self.og_dataset = self.dataset
    
        # map class labels into multi-one-hot representations
        
        if self.inference:
            self.dataset = self.dataset.loc[:, ['peptide', 'hla', 'assay_mhc_type', 'assay_method', 'assay_technique',
                                                'normalized_label']]
            
        else:
            if classes is not None:
                self.dataset['normalized_label'] = self.dataset.loc[:, 'assay_qualitative_measure']\
                                                    .apply(self.get_multi_one_hot)
                self.dataset = self.dataset.loc[:, ['peptide', 'hla', 'assay_mhc_type', 'assay_method', 'assay_technique',
                                                    'assay_measurement_inequality', 'normalized_label']]
        
            else:
                self.dataset = self.dataset.groupby(['peptide',
                                                    'hla',
                                                    'assay_mhc_type',
                                                    'assay_method',
                                                    'assay_technique',
                                                    'assay_measurement_inequality'])['normalized_label'].median().reset_index()
        
            # map inequalities
            ineq_mapping = {'>': -1, '=': 0, '<': 1}
            self.dataset['assay_measurement_inequality_number'] = self.dataset['assay_measurement_inequality'].map(ineq_mapping)

        # # peptide features
        # pca_path = f'_npca={pca_components}' if pca_components else 'no_pca'

        # # create processed folder if it doesnt exist
        # processed_path = f'{root_folder}/data/processed/affinity'
        # Path(processed_path).mkdir(parents=True, exist_ok=True)
        # peptide_features_path = f'{processed_path}/{data_filename.split(".")[0]}_peptide_features{pca_path}_{split}_{fold}.pkl'

        # # if the file exists a we dont overwrite, load features
        # if os.path.exists(peptide_features_path) and overwrite is False:
        #     print_color('Peptide features file exists, loading.', 'OKGREEN')
        #     self.peptide_features = pickle.load(open(peptide_features_path, 'rb'))
        # else:
        #     print_color('Creating peptide features', 'OKBLUE')
        #     # create matrix [n_aminoacids, aminoacid_features] for each peptide
        #     self.peptide_features = {}
        #     for p in tqdm(self.dataset.peptide.unique()):
        #         feats = np.zeros((peptide_max_len, self.aminoacid_features.shape[1]))
        #         feats[:len(p)] = [self.aminoacid_features.loc[aa] for aa in p]
        #         self.peptide_features[p] = feats
        #     pickle.dump(self.peptide_features, open(peptide_features_path, 'wb'))

        # # HLA features
        
        # hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_representations_{split}_{fold}.pkl'
        # if msa:
        #     hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_msa_representations_{split}_{fold}.pkl'

        # # if the file exists and not overwrite flag, load features
        # if os.path.exists(hla_features_path) and overwrite is False:
        #     print_color('Loading HLA features', 'OKGREEN')
        #     self.hla_features = pickle.load(open(hla_features_path, 'rb'))
        # else:
        #     print_color('Creating HLA features', 'OKGREEN')
        #     # load HLA sequences and get pre-trained representations
        #     hla_sequences_path = f'./data/{hla_sequences_filename}'
        #     assert os.path.exists(hla_sequences_path), 'No HLA sequences file found.'
        #     hla_sequences = pd.read_csv(hla_sequences_path)

        #     # Load ESM-1b model
        #     if msa:
        #         model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        #     else:
        #         model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        #     model = model.to('cuda')
        #     # get features and dump
        #     # self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet)
        #     self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet, msa=msa)
        #     pickle.dump(self.hla_features, open(hla_features_path, 'wb'))

        """
        TAPE data processing
        """
        tokenizer = TAPETokenizer(vocab='iupac')
        self.tokenizer = tokenizer

        hla_sequences_path = f'./data/{hla_sequences_filename}'
        assert os.path.exists(hla_sequences_path), 'No HLA sequences file found.'
        self.hla_sequences = pd.read_csv(hla_sequences_path).set_index('hla')
        
        # assert if sklearn OneHotEncoder is properly fitted.
        if self.assay_onehot_encoder is not None:
            for i, assay_feat in enumerate(['assay_mhc_type', 'assay_method', 'assay_technique']):
                # assert set(self.assay_onehot_encoder.categories_[i]) == \
                #         set(self.dataset.loc[:,assay_feat].values)
                for j, value in enumerate(set(self.dataset.loc[:,assay_feat].values)):
                    assert value in set(self.assay_onehot_encoder.categories_[i])
                    
    def get_multi_one_hot(self, cls):
        '''
            Order of classes should be descending
        '''
        for i in range(len(self.classes)):
            if cls == self.classes[i]:
                return [0]*i + [1]*(len(self.classes)-i-1)
                # return [1]*(len(self.classes)-i-1) + [0]*i

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        
        peptide, hla_name, assay, method, technique, symbol, label, inequality = \
            self.dataset.iloc[idx][['peptide', 'hla', 'assay_mhc_type', 'assay_method', 'assay_technique',
                    'assay_measurement_inequality', 'normalized_label', 'assay_measurement_inequality_number']].values

        assay_features = None

        hla = self.hla_sequences.loc[hla_name, 'seq']
        input_ids = self.tokenizer.encode(hla+peptide)
        input_mask = np.ones_like(input_ids)
        items = {'hla_name': hla_name,
                'peptide_name': peptide,
                # 'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
                # 'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
                'input_ids': input_ids,
                'input_mask': input_mask,
                'reg_target': torch.tensor(label).float(),
                # 'assay_features': assay_features,
                'inequality': torch.tensor(inequality).double(),
                'inequality_symbol': symbol,
                }

        if self.use_assay_features:
            # get assay features as onehot vector
            
            assay_features = self.assay_onehot_encoder.transform([[assay, method, technique]]).toarray()
            assay_features = torch.from_numpy(assay_features).float()
            items['assay_features'] = assay_features
        
        return items
        

        # return {'hla_name': hla_name,
        #         'peptide_name': peptide,
        #         'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
        #         'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
        #         'reg_target': torch.tensor(label).float(),
        #         'assay_features': assay_features,
        #         'inequality': torch.tensor(inequality).double(),
        #         'inequality_symbol': symbol}

    def collate_fn(self, batch):
        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}
        items = {'hla_name': batch['hla_name'],
                'peptide_name': batch['peptide_name'],
                'input_ids': torch.from_numpy(tape_pad(batch['input_ids'], 0)),
                'input_mask': torch.from_numpy(tape_pad(batch['input_mask'], 0)),
                'reg_target': torch.stack(batch['reg_target']),
                # 'assay_features': torch.stack(batch['assay_features']),
                'inequality': torch.stack(batch['inequality']),
                'inequality_symbol': batch['inequality_symbol']
        }
        if self.use_assay_features:
            items['assay_features'] = torch.stack(batch['assay_features'])
        return items





class ImmunogenicityDataset(Dataset):
    def __init__(self,
                 dataset,
                 data_filename,
                 root_folder,
                 peptide_max_len,
                 use_blosum,
                 use_process_features,
                 process_onehot_encoder,
                 msa,
                 pca_components,
                 hla_sequences_filename,
                 split,
                 fold,
                 task,
                 use_weight_loss,
                 overwrite):
        """
        Torch dataset for immunogenicity task. Generates features for each HLA and peptide sequence
        using AAindex (aminoacid level), BLOSUM62 (aminoacid level) matrix and assay features (sequence level).
        :param dataset: dataset as tabular date. [pd.DataFrame]
        :param data_filename: dataset filename. [str]
        :param root_folder: path of the root folder [str]
        :param peptide_max_len: longest peptide sequence in the dataset. [str]
        :param use_blosum: wheter concatenate BLOSUM62 matrix features to inputs. [bool]
        :param pca_components: number of AAindex principal components to use. [int]
        :param hla_sequences_filename: hla-name to hla-sequence filename. [str]
        :param split: used for saving preprocessed files. [str]
        :param fold: used for saving preprocessed files. Assumes pre-generated folds [str]
        :param overwrite: wether to overwrite preprocessed files (if exist). [bool]
        """
        self.root_folder = root_folder
        self.task = task
        self.use_weight_loss = use_weight_loss
        self.use_process_features = use_process_features
        self.process_onehot_encoder = process_onehot_encoder
        self.peptide_max_len = peptide_max_len
        self.use_blosum = use_blosum
        self.pca_components = pca_components
        self.split = split

        # deepcopy to avoid problems with dataframe in torch lightning data module
        self.dataset = dataset.copy(deep=True)

        # aminoacid features and normalization
        self.aminoacid_features = pd.read_csv(f'{root_folder}/data/aminoacid_features.csv', index_col='aminoacid')

        # pca projection for aminoacid features (AAIndex)
        self.pca = PCA(pca_components)
        self.aminoacid_features = pd.DataFrame(self.pca.fit_transform(self.aminoacid_features.values),
                                               columns=[f'pca_{i}' for i in range(pca_components)],
                                               index=self.aminoacid_features.index)

        # get blossum matrix and attach to aminoacid faetures
        if self.use_blosum:
            # TODO: test thing part for bugs, merge might introduce some undesired behaviour
            self.blosum_features = get_blosum62_matrix()

            # take only columns of blosum matrix present in the index of the aminoacid_features dataframe
            self.aminoacid_features = pd.merge(self.blosum_features[self.aminoacid_features.index],
                                               self.aminoacid_features,
                                               left_index=True,
                                               right_index=True)

        # normalize features
        self.amino_scaler = StandardScaler()
        self.aminoacid_features.loc[:, :] = self.amino_scaler.fit_transform(self.aminoacid_features)

        # ignore samples with aminoacids outside of 20 essential
        allowed_vocabulary = set(self.aminoacid_features.index)
        self.dataset.loc[:, 'peptide'] = self.dataset.peptide.str.upper()
        self.dataset.peptide = self.dataset.peptide.str.replace('-', '')
        self.dataset = self.dataset[self.dataset.peptide.apply(lambda x: set(x).issubset(allowed_vocabulary))]

        self.og_dataset = self.dataset

        # create processed folder if it doesnt exist
        processed_path = f'{root_folder}/data/processed/immunogenicity'
        Path(processed_path).mkdir(parents=True, exist_ok=True)

        # peptide features
        pca_path = f'_npca={pca_components}' if pca_components else 'no_pca'
        peptide_features_path = f'{processed_path}/{data_filename.split(".")[0]}_peptide_features{pca_path}_{split}_{fold}.pkl'

        # if the file exists a we dont overwrite, load features
        if os.path.exists(peptide_features_path) and overwrite is False:
            print_color('Peptide features file exists, loading.', 'OKGREEN')
            self.peptide_features = pickle.load(open(peptide_features_path, 'rb'))
        else:
            print_color('Creating peptide features', 'OKBLUE')
            # create matrix [n_aminoacids, aminoacid_features] for each peptide
            self.peptide_features = {}
            for p in tqdm(self.dataset.peptide.unique()):
                feats = np.zeros((peptide_max_len, self.aminoacid_features.shape[1]))
                feats[:len(p)] = [self.aminoacid_features.loc[aa] for aa in p]
                self.peptide_features[p] = feats
            pickle.dump(self.peptide_features, open(peptide_features_path, 'wb'))

        # HLA features
        hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_representations_{split}_{fold}.pkl'
        if msa:
            hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_msa_representations_{split}_{fold}.pkl'

        # if the file exists a we dont overwrite, load features
        if os.path.exists(hla_features_path) and overwrite is False:
            print_color('Loading HLA features', 'OKGREEN')
            self.hla_features = pickle.load(open(hla_features_path, 'rb'))
        else:
            print_color('Creating HLA features', 'OKGREEN')
            # load HLA sequences and get pre-trained representations
            hla_sequences_path = f'./data/{hla_sequences_filename}'
            assert os.path.exists(hla_sequences_path), 'No HLA sequences file found.'
            hla_sequences = pd.read_csv(hla_sequences_path)

            # Load ESM-1b model
            if msa:
                model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            else:
                model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            model = model.to('cuda')
            # get features and dump
            # self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet)
            self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet, msa=msa)
            pickle.dump(self.hla_features, open(hla_features_path, 'wb'))

        # calculate loss function sample weights depending on the variance.
        # values with less variance should be given a higher loss.
        # 0.8 factor help avoiding samples with 0 weight due to the min-max normalization.
        if self.use_weight_loss:
            self.dataset['sample_loss_weight'] = 1 / self.dataset.variance
            self.dataset['sample_loss_weight'] = (self.dataset.sample_loss_weight - self.dataset.sample_loss_weight.min() * 0.8) / \
                                                (self.dataset.sample_loss_weight.max() - self.dataset.sample_loss_weight.min())

        # # output onehot representations of peptide as well
        # self.amino_acids = pd.read_csv('data/aminoacid_features.csv').sort_values('aminoacid').reset_index(drop=True)
        
        # self.amino2idx = {}
        # # self.idx2amino = {}
        # for i in range(len(self.amino_acids)):
        #     # self.amino2idx[self.amino_acids.loc[i, 'aminoacid']] = i+1
        #     # self.idx2amino[i+1] = self.amino_acids.loc[i, 'aminoacid']
        #     vector = np.zeros(len(self.amino_acids)+1)
        #     vector[i+1] = 1.
        #     self.amino2idx[self.amino_acids.loc[i, 'aminoacid']] = vector
        #     # self.idx2amino[i+1] = self.amino_acids.loc[i, 'aminoacid']
        
        # # self.amino2idx['#'] = 0
        # # self.idx2amino[0] = '#'
        # self.amino2idx['#'] = np.zeros(len(self.amino_acids)+1)
        # self.amino2idx['#'][0] = 1.
        
        # # map inequalities
        # # ineq_mapping = {'>': -1, '=': 0, '<': 1} # w.r.t. inequality symbol from IC50
        # ineq_mapping = {'>': 1, '=': 0, '<': -1} # w.r.t. inequality symbol from Immunogenicity label
        # self.dataset['assay_measurement_inequality_number'] = self.dataset['assay_measurement_inequality'].map(ineq_mapping)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        # # hla,peptide,test,respond,label,expected_value,variance,reg_target,fold
        # hla_name, peptide, respond, test = self.dataset.iloc[idx][['hla', 'peptide', 'respond', 'test']].values
        # reg_target, sample_loss_weight = self.dataset.iloc[idx][['reg_target', 'sample_loss_weight']].values

        # return {'hla_name': hla_name,
        #         'peptide_name': peptide,
        #         'respond': respond,
        #         'test': test,
        #         'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
        #         'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
        #         'sample_loss_weight': torch.tensor(sample_loss_weight).float(),
        #         'reg_target': torch.tensor(reg_target).float()}
        
        hla_name, peptide = self.dataset.iloc[idx][['hla', 'peptide']].values
        reg_target = self.dataset.iloc[idx]['reg_target']
        
        output_dict = {'hla_name': hla_name,
                    'peptide_name': peptide,
                    'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
                    'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
                    'reg_target': torch.tensor(reg_target).float()}


        if self.task == 'immunogenicity':
            respond, test, expected_value, variance = self.dataset.iloc[idx]\
                                                    [['respond', 'test', 'expected_value', 'variance']].values
            output_dict['respond'] = respond
            output_dict['test'] = test
            output_dict['expected_value'] = expected_value
            output_dict['variance'] = variance
            
            # if self.split == 'train':
            #     """
            #     Augmentation session
            #     based on 
            #     augment by probability p, noise added by variance of the instance
            #     """
            #     aug_p = .5
            #     if np.random.uniform()>aug_p:

            #         noise = np.zeros((self.peptide_max_len, self.pca_components))
            #         if self.use_blosum:
            #             noise = np.zeros((self.peptide_max_len, self.pca_components+20))
            #         for i, pep in enumerate(peptide):
                        
                        
            #             # std_aa = (minimum_diffs_peptide[aa]/2)*variance
            #             std_aa = np.fromiter(minimum_diffs_peptide.values(), dtype=float)*variance
            #             noise_aa = np.random.normal(scale=std_aa)
            #             noise[i] = noise_aa
                        
            #         noise = torch.FloatTensor(noise)
            #         # print(self.peptide_features[peptide].shape)
            #         # print(noise.shape)
            #         output_dict['peptide'] = torch.from_numpy(self.peptide_features[peptide]).float()\
            #                                     + noise
                    
            #         # Inequality label for ranged mse loss
            #         output_dict['inequality'] = torch.ones(1)
            #     else:
            #         output_dict['inequality'] = torch.zeros(1)
            
        if self.use_weight_loss:
            sample_loss_weight = self.dataset.iloc[idx]['sample_loss_weight']
            output_dict['sample_loss_weight'] = torch.tensor(sample_loss_weight).float()

        if self.use_process_features:
            # get process features as onehot vector
            process = self.dataset.iloc[idx]['in_vitro_process_type']
            process_features = self.process_onehot_encoder.transform([[process]]).toarray()
            output_dict['process_features'] = torch.from_numpy(process_features).float()

        # Use Inequality loss for augmented immunogenicity dataset
        symbol, inequality = \
            self.dataset.iloc[idx][['assay_measurement_inequality', 'assay_measurement_inequality_number']].values
        output_dict['inequality'] = torch.tensor(inequality).double()
        output_dict['inequality_symbol'] = symbol

        # # output onehot representations of peptide as well
        # output_dict['peptide_onehot'] = torch.from_numpy(np.stack(
        #                 [[self.amino2idx[y] for y in x] + \
        #                     [self.amino2idx['#']]*(self.peptide_max_len-len(x))
        #                     # [self.amino2idx['#']]*(self.dataset_params['peptide_max_len']-len(x)+1)
        #                     for x in peptide]))

        return output_dict



class ElutedDataset(Dataset):
    def __init__(self,
                 dataset,
                 data_filename,
                 root_folder,
                 peptide_max_len,
                 use_blosum,
                 use_process_features,
                 process_onehot_encoder,
                 msa,
                 pca_components,
                 hla_sequences_filename,
                 split,
                 fold,
                 task,
                 use_weight_loss,
                 overwrite):
        """
        Torch dataset for immunogenicity task. Generates features for each HLA and peptide sequence
        using AAindex (aminoacid level), BLOSUM62 (aminoacid level) matrix and assay features (sequence level).
        :param dataset: dataset as tabular date. [pd.DataFrame]
        :param data_filename: dataset filename. [str]
        :param root_folder: path of the root folder [str]
        :param peptide_max_len: longest peptide sequence in the dataset. [str]
        :param use_blosum: wheter concatenate BLOSUM62 matrix features to inputs. [bool]
        :param pca_components: number of AAindex principal components to use. [int]
        :param hla_sequences_filename: hla-name to hla-sequence filename. [str]
        :param split: used for saving preprocessed files. [str]
        :param fold: used for saving preprocessed files. Assumes pre-generated folds [str]
        :param overwrite: wether to overwrite preprocessed files (if exist). [bool]
        """
        self.root_folder = root_folder
        self.task = task
        self.use_weight_loss = use_weight_loss
        self.use_process_features = use_process_features
        self.process_onehot_encoder = process_onehot_encoder
        self.peptide_max_len = peptide_max_len

        # deepcopy to avoid problems with dataframe in torch lightning data module
        self.dataset = dataset.copy(deep=True)

        # aminoacid features and normalization
        self.aminoacid_features = pd.read_csv(f'{root_folder}/data/aminoacid_features.csv', index_col='aminoacid')

        # pca projection for aminoacid features (AAIndex)
        self.pca = PCA(pca_components)
        self.aminoacid_features = pd.DataFrame(self.pca.fit_transform(self.aminoacid_features.values),
                                               columns=[f'pca_{i}' for i in range(pca_components)],
                                               index=self.aminoacid_features.index)

        # get blossum matrix and attach to aminoacid faetures
        if use_blosum:
            # TODO: test thing part for bugs, merge might introduce some undesired behaviour
            blosum_features = get_blosum62_matrix()

            # take only columns of blosum matrix present in the index of the aminoacid_features dataframe
            self.aminoacid_features = pd.merge(blosum_features[self.aminoacid_features.index],
                                               self.aminoacid_features,
                                               left_index=True,
                                               right_index=True)

        # normalize features
        self.amino_scaler = StandardScaler()
        self.aminoacid_features.loc[:, :] = self.amino_scaler.fit_transform(self.aminoacid_features)

        # ignore samples with aminoacids outside of 20 essential
        allowed_vocabulary = set(self.aminoacid_features.index)
        self.dataset.loc[:, 'peptide'] = self.dataset.peptide.str.upper()
        self.dataset.peptide = self.dataset.peptide.str.replace('-', '')
        self.dataset = self.dataset[self.dataset.peptide.apply(lambda x: set(x).issubset(allowed_vocabulary))]

        self.og_dataset = self.dataset

        # create processed folder if it doesnt exist
        processed_path = f'{root_folder}/data/processed/immunogenicity'
        Path(processed_path).mkdir(parents=True, exist_ok=True)

        # peptide features
        pca_path = f'_npca={pca_components}' if pca_components else 'no_pca'
        peptide_features_path = f'{processed_path}/{data_filename.split(".")[0]}_peptide_features{pca_path}_{split}_{fold}.pkl'

        # if the file exists a we dont overwrite, load features
        if os.path.exists(peptide_features_path) and overwrite is False:
            print_color('Peptide features file exists, loading.', 'OKGREEN')
            self.peptide_features = pickle.load(open(peptide_features_path, 'rb'))
        else:
            print_color('Creating peptide features', 'OKBLUE')
            # create matrix [n_aminoacids, aminoacid_features] for each peptide
            self.peptide_features = {}
            for p in tqdm(self.dataset.peptide.unique()):
                feats = np.zeros((peptide_max_len, self.aminoacid_features.shape[1]))
                feats[:len(p)] = [self.aminoacid_features.loc[aa] for aa in p]
                self.peptide_features[p] = feats
            pickle.dump(self.peptide_features, open(peptide_features_path, 'wb'))

        # HLA features
        hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_representations_{split}_{fold}.pkl'
        if msa:
            hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_msa_representations_{split}_{fold}.pkl'

        # if the file exists a we dont overwrite, load features
        if os.path.exists(hla_features_path) and overwrite is False:
            print_color('Loading HLA features', 'OKGREEN')
            self.hla_features = pickle.load(open(hla_features_path, 'rb'))
        else:
            print_color('Creating HLA features', 'OKGREEN')
            # load HLA sequences and get pre-trained representations
            hla_sequences_path = f'./data/{hla_sequences_filename}'
            assert os.path.exists(hla_sequences_path), 'No HLA sequences file found.'
            hla_sequences = pd.read_csv(hla_sequences_path)

            # Load ESM-1b model
            if msa:
                model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            else:
                model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            model = model.to('cuda')
            # get features and dump
            # self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet)
            self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet, msa=msa)
            pickle.dump(self.hla_features, open(hla_features_path, 'wb'))

        # calculate loss function sample weights depending on the variance.
        # values with less variance should be given a higher loss.
        # 0.8 factor help avoiding samples with 0 weight due to the min-max normalization.
        if self.use_weight_loss:
            self.dataset['sample_loss_weight'] = 1 / self.dataset.variance
            self.dataset['sample_loss_weight'] = (self.dataset.sample_loss_weight - self.dataset.sample_loss_weight.min() * 0.8) / \
                                                (self.dataset.sample_loss_weight.max() - self.dataset.sample_loss_weight.min())

        # # output onehot representations of peptide as well
        # self.amino_acids = pd.read_csv('data/aminoacid_features.csv').sort_values('aminoacid').reset_index(drop=True)
        
        # self.amino2idx = {}
        # # self.idx2amino = {}
        # for i in range(len(self.amino_acids)):
        #     # self.amino2idx[self.amino_acids.loc[i, 'aminoacid']] = i+1
        #     # self.idx2amino[i+1] = self.amino_acids.loc[i, 'aminoacid']
        #     vector = np.zeros(len(self.amino_acids)+1)
        #     vector[i+1] = 1.
        #     self.amino2idx[self.amino_acids.loc[i, 'aminoacid']] = vector
        #     # self.idx2amino[i+1] = self.amino_acids.loc[i, 'aminoacid']
        
        # # self.amino2idx['#'] = 0
        # # self.idx2amino[0] = '#'
        # self.amino2idx['#'] = np.zeros(len(self.amino_acids)+1)
        # self.amino2idx['#'][0] = 1.

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        # # hla,peptide,test,respond,label,expected_value,variance,reg_target,fold
        # hla_name, peptide, respond, test = self.dataset.iloc[idx][['hla', 'peptide', 'respond', 'test']].values
        # reg_target, sample_loss_weight = self.dataset.iloc[idx][['reg_target', 'sample_loss_weight']].values

        # return {'hla_name': hla_name,
        #         'peptide_name': peptide,
        #         'respond': respond,
        #         'test': test,
        #         'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
        #         'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
        #         'sample_loss_weight': torch.tensor(sample_loss_weight).float(),
        #         'reg_target': torch.tensor(reg_target).float()}
        
        hla_name, peptide = self.dataset.iloc[idx][['hla', 'peptide']].values
        reg_target = self.dataset.iloc[idx]['reg_target']
        source = self.dataset.iloc[idx]['source']
        
        output_dict = {'hla_name': hla_name,
                    'peptide_name': peptide,
                    'source': source,
                    'peptide': torch.from_numpy(self.peptide_features[peptide]).float(),
                    'hla': torch.from_numpy(self.hla_features[hla_name]).float(),
                    'reg_target': torch.tensor(reg_target).float()}
        if self.task == 'immunogenicity':
            respond, test = self.dataset.iloc[idx][['respond', 'test']].values
            output_dict['respond'] = respond
            output_dict['test'] = test
            
        if self.use_weight_loss:
            sample_loss_weight = self.dataset.iloc[idx]['sample_loss_weight']
            output_dict['sample_loss_weight'] = torch.tensor(sample_loss_weight).float()

        if self.use_process_features:
            # get process features as onehot vector
            process = self.dataset.iloc[idx]['in_vitro_process_type']
            process_features = self.process_onehot_encoder.transform([[process]]).toarray()
            output_dict['process_features'] = torch.from_numpy(process_features).float()

        # # output onehot representations of peptide as well
        # output_dict['peptide_onehot'] = torch.from_numpy(np.stack(
        #                 [[self.amino2idx[y] for y in x] + \
        #                     [self.amino2idx['#']]*(self.peptide_max_len-len(x))
        #                     # [self.amino2idx['#']]*(self.dataset_params['peptide_max_len']-len(x)+1)
        #                     for x in peptide]))

        return output_dict
