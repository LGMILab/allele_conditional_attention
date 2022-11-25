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

from ems.generate_esm import get_esm_representations, get_peptide_representations
# from commons.augmentation import mhc_i_imm_aug

from tape.tokenizers import TAPETokenizer
from tape.datasets import pad_sequences as tape_pad


"""
TransPHLA hyperparameters
"""

pep_max_len = 14 # peptide; enc_input max sequence length
# hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
hla_max_len = 181 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len

vocab = np.load('data/transphla/vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)

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
        self.hla_max_len = dataset_params['hla_max_len']
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
        # self.use_esm = dataset_params['use_esm']
        # self.use_aa = dataset_params['use_aa']
        # self.use_pep_esm = dataset_params['use_pep_esm']
        self.emb_type = dataset_params['emb_type']
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
                                           hla_max_len=self.hla_max_len,
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
                                           emb_type=self.emb_type,
                                           inference=False)

        print('number of Train dataset instances : '+str(len(self.train_split)))

        # validation dataset
        self.val_split = AffinityDataset(dataset=val_dataset,
                                         data_filename=self.data_filename,
                                         root_folder=self.data_dir,
                                         peptide_max_len=self.peptide_max_len,
                                         hla_max_len=self.hla_max_len,
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
                                         emb_type=self.emb_type,                                        
                                         inference=False)
        print('number of Valid dataset instances : '+str(len(self.val_split)))

        # if there is no test data use validation split as test
        if self.test_data_filename is None:
            self.test_split = deepcopy(self.val_split)
        else:
            self.test_split = AffinityDataset(dataset=self.test_dataset,
                                                data_filename=self.test_data_filename,
                                                root_folder=self.data_dir,
                                                peptide_max_len=self.peptide_max_len,
                                                hla_max_len=self.hla_max_len,
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
                                                # use_esm=self.use_esm,
                                                # use_aa=self.use_aa,
                                                # use_pep_esm=self.use_pep_esm,
                                                emb_type=self.emb_type,
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
                          num_workers=self.num_workers, pin_memory=True, sampler=None)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, shuffle=False, 
                          pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False, 
                          pin_memory=True, num_workers=self.num_workers)


class AffinityDataset(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame,
                 data_filename: str,
                 root_folder: str,
                 peptide_max_len: int,
                 hla_max_len: int,
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
                #  use_esm: bool,
                #  use_aa: bool,
                #  use_pep_esm: bool,
                 emb_type: str,
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

        # self.use_esm = use_esm
        # self.use_aa = use_aa
        # self.use_pep_esm = use_pep_esm
        self.emb_type = emb_type
        self.peptide_max_len = peptide_max_len
        self.hla_max_len = hla_max_len
        
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


        # if self.use_esm:
        if self.emb_type in ['aa+esm', 'aa2']:
            # create processed folder if it doesnt exist
            processed_path = f'{root_folder}/data/processed/affinity'
            Path(processed_path).mkdir(parents=True, exist_ok=True)

            # if self.use_pep_esm:
            if self.emb_type == 'esm2':
                peptide_features_path = f'{root_folder}/data/processed/affinity/mhcI_esm1b_peptide_representations.pkl'
                if os.path.exists(peptide_features_path) and overwrite is False:
                    print_color('Peptide features file exists, loading.', 'OKGREEN')
                    self.peptide_features = pickle.load(open(peptide_features_path, 'rb'))
                else:
                    print_color('Creating peptide features', 'OKBLUE')
                    
                    # Load ESM-1b model
                    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                    model = model.to('cuda')
                    # get features and dump
                    self.peptide_features = get_peptide_representations(model, dataset, alphabet)
                    pickle.dump(self.peptide_features, open(peptide_features_path, 'wb'))


            else:
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

            # if the file exists and not overwrite flag, load features
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
                model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                model = model.to('cuda')
                # get features and dump
                self.hla_features = get_esm_representations(model, dataset, hla_sequences, alphabet)
                pickle.dump(self.hla_features, open(hla_features_path, 'wb'))
                del model
                del alphabet
        
        # if self.use_aa:
        if self.emb_type == 'aa2':
        
            # create processed folder if it doesnt exist
            processed_path = f'{root_folder}/data/processed/affinity'
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
            
            hla_features_path = f'{processed_path}/{data_filename.split(".")[0]}_hla_aa_representations_{split}_{fold}.pkl'
            
            # if the file exists and not overwrite flag, load features
            if os.path.exists(hla_features_path) and overwrite is False:
                print_color('Loading HLA features', 'OKGREEN')
                self.hla_features = pickle.load(open(hla_features_path, 'rb'))
            else:
                
                print_color('Creating HLA features', 'OKGREEN')
                # load HLA sequences and get pre-trained representations
                hla_sequences_path = f'./data/{hla_sequences_filename}'
                assert os.path.exists(hla_sequences_path), 'No HLA sequences file found.'
                hla_sequences = pd.read_csv(hla_sequences_path)
                
                self.hla_features = {}
                
                for idx in tqdm(hla_sequences.index):
                    feats = np.zeros((hla_max_len, self.aminoacid_features.shape[1]))
                    feats[:len(hla_sequences.loc[idx, 'seq'])] = [self.aminoacid_features.loc[aa] 
                                                            for aa in hla_sequences.loc[idx, 'seq']]
                    self.hla_features[hla_sequences.loc[idx, 'hla']] = feats
                    
                pickle.dump(self.hla_features, open(hla_features_path, 'wb'))

        elif self.emb_type == 're':
            
            # Acquire list of hla sequences
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
        # peptide, hla_name, assay, method, technique, symbol, label, inequality = self.dataset.iloc[idx].values
        
        items = {}

        if self.inference:
            peptide, hla_name, assay, method, technique = \
                self.dataset.iloc[idx][['peptide', 'hla', 'assay_mhc_type', 'assay_method', 'assay_technique']].values
            
            items['hla_name'] = hla_name
            items['peptide_name'] = peptide
            items['peptide'] = torch.from_numpy(self.peptide_features[peptide]).float()
            items['hla'] = torch.from_numpy(self.hla_features[hla_name]).float()

            assay_features = None
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
            #         'assay_features': assay_features,
            #         # 'inequality': torch.tensor(inequality).double()
            #         }
                
        else:
            peptide, hla_name, assay, method, technique, symbol, label, inequality = \
                self.dataset.iloc[idx][['peptide', 'hla', 'assay_mhc_type', 'assay_method', 'assay_technique',
                        'assay_measurement_inequality', 'normalized_label', 'assay_measurement_inequality_number']].values

            if self.emb_type == 're':
                """
                TransPHLA data processing
                """
                hla = self.hla_sequences.loc[hla_name, 'seq']

                pep, hla = peptide.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
                pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
                hla_input = [[vocab[n] for n in hla]]

            items['hla_name'] = hla_name
            items['peptide_name'] = peptide
            if self.emb_type == 're':
                items['peptide'] = torch.LongTensor(pep_input).squeeze()
                items['hla'] = torch.LongTensor(hla_input).squeeze()
            else:
                items['peptide'] = torch.from_numpy(self.peptide_features[peptide]).float()
                items['hla'] = torch.from_numpy(self.hla_features[hla_name]).float()
            items['reg_target'] = torch.tensor(label).float()
            items['inequality'] = torch.tensor(inequality).double()
            items['inequality_symbol'] = symbol

            assay_features = None
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
        """
        TAPE data processing
        """
        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}
        items = {'hla_name': batch['hla_name'],
                'peptide_name': batch['peptide_name'],
                # 'input_ids': torch.from_numpy(tape_pad(batch['input_ids'], 0)),
                # 'input_mask': torch.from_numpy(tape_pad(batch['input_mask'], 0)),
                'peptide': torch.from_numpy(tape_pad(batch['peptide'], 0)),
                'peptide_mask': torch.from_numpy(tape_pad(batch['peptide_mask'], 0)),
                'hla': torch.from_numpy(tape_pad(batch['hla'], 0)),
                'hla_mask': torch.from_numpy(tape_pad(batch['hla_mask'], 0)),
                'reg_target': torch.stack(batch['reg_target']),
                # 'assay_features': torch.stack(batch['assay_features']),
                'inequality': torch.stack(batch['inequality']),
                'inequality_symbol': batch['inequality_symbol']
        }
        if self.use_assay_features:
            items['assay_features'] = torch.stack(batch['assay_features'])
        return items
