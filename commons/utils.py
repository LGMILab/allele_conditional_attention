import copy
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import yaml
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SubsMat import MatrixInfo as matlist
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from scipy.stats import betabinom
import math

# set of colors for verbose prints.
bcolors = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKCYAN': '\033[96m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'}

def get_default_assay():
    # ['purified MHC', 'competitive', 'fluorescence']
    return torch.Tensor([[1., 0., 0., 1., 1., 0.]])

def normalize(x):
    return 1-math.log(x)/math.log(50000)

def reverse_normalize(y):
    '''
    y = 1-log(x)/log(50000)
    x = exp(log(50000)*(1-y))
    '''
    # return math.exp(math.log(50000)*(1-y))
    return min(math.exp(math.log(50000)*(1-y)), 50000.)

def print_color(p, color):
    """
    helper function to print string with certain style.
    :param p: string to print. [str]
    :param color: one of the predefined colors in bcolors. [str]
                  {'HEADER', 'OKBLUE', 'OKCYAN', 'OKGREEN', 'WARNING',
                   'FAIL', 'ENDC', 'BOLD', 'UNDERLINE'}
    """
    if color not in bcolors.keys():
        raise ValueError(f'This color code does not exists. Chose between {bcolors.keys()}')
    print(bcolors[color] + bcolors['BOLD'] + p + bcolors['ENDC'] + bcolors['ENDC'])
    return

def aminoacid_sequence_tokenizer(aa, max_length=10):
    """
    Tokenizer for peptides and HLA sequences.
    :param aa: aminoacid sequence. [str]
    :param max_length: max sequence lenght in dataset. [int]
    :return: tokenized version. [list[str]]
    """
    tokens = list(aa)
    if len(aa) != max_length:
        tokens += ['<PAD>'] * (max_length - len(aa))
    return tokens

def get_blosum62_matrix():
    """
    Get BLOSUM62 in dense matrix form from sparse matrix notation of Biopython.
    :return: [Pandas DataFrame].
    """

    # get unique letters
    unique_aminoacids = set([a[0] for a in matlist.blosum62.keys()])

    # assign a number to each letter to fill dense matrix
    unique_aminoacids = {k: v for k, v in zip(sorted(list(unique_aminoacids)), range(len(unique_aminoacids)))}

    # create zeroes matrix and populate it (lower matrix -> add upper -> remove diagonal)
    blosum62 = np.zeros((len(unique_aminoacids), len(unique_aminoacids)))
    for k, v in matlist.blosum62.items():
        blosum62[unique_aminoacids[k[0]], unique_aminoacids[k[1]]] = v
    blosum62 = blosum62 + blosum62.T - np.eye(len(unique_aminoacids)) * np.diag(blosum62)

    # return as dataframe
    blosum62 = pd.DataFrame(blosum62.astype(int), index=unique_aminoacids.keys(), columns=unique_aminoacids.keys())

    return blosum62

def create_folds_for_hla_list(data: pd.DataFrame,
                              n_folds: int = 5,
                              hlas=None):
    """
    Create balanced train-val folds for the alleles of interest, while also splitting exprerimental samples
    of each peptide (could be same HLA or differnet) as a block, i.e all data of peptide in a particular
    publication goes to either train or validation as block. This tries to avoid overfitting on samples
    that come from the same author (which might have a similar bias), and would make it easier to predict
    validation data by just looking at measurement from the same paper.
    :param data: data in tabular format. [pd.DataFrame]
    :param n_folds: number of folds. [int]
    :param hlas: hlas of interest to split equally across folds. list[str]
    :return: original dataframe with a column name 'fold', which contains the splits. [pd.DataFrame]
    """
    if hlas is None:
        hlas = ['HLA-A*0201', 'HLA-A*2402']
    data['fold'] = -1

    # per hla samples
    for h in hlas:
        # get all papers that contain hla
        for paper_name, n_samples in data[data.hla == h].reference_title.value_counts().items():

            # get paper data
            d = data[(data.hla == h) & (data.reference_title == paper_name)]

            # get unique peptides
            unique_peptides = d.peptide.unique()

            # if there is not enough unique peptides, no need to split them as block
            if unique_peptides.shape[0] > n_folds:

                # shuffle data
                np.random.shuffle(unique_peptides)

                # split in n folds and assing a different fold to each
                sampled_peptides = np.array_split(unique_peptides, n_folds)
                for fold, peps in enumerate(sampled_peptides):
                    data.loc[(data.hla == h) & (data.reference_title == paper_name) & (data.peptide.isin(peps)), 'fold'] = fold

    # add a random sample of equal size to each fold.
    for i in range(n_folds):
        s = data[data.fold == -1].sample(data[data.fold == i].shape[0]).index
        data.loc[s, 'fold'] = i
    return data


def load_yml_config(path: str, previous_includes: list = None):
    """
    Adapted from https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/common/utils.py
    :param path: config YML file path. [str]
    :param previous_includes: other files to include. [list]
    :return:
    """
    if not previous_includes:
        previous_includes = []

    path = Path(path)
    if path in previous_includes:
        raise ValueError(f"Cyclic config include detected. {path} included in sequence {previous_includes}.")
    previous_includes = previous_includes + [path]

    direct_config = yaml.safe_load(open(path, "r"))

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError("Includes must be a list, '{}' provided".format(type(includes)))

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_yml_config(include, previous_includes)
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def merge_dicts(dict1: dict, dict2: dict):
    """
    Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    :param dict1: first dict. [dict]
    :param dict2: values in dict2 will override values from dict1 in case they share the same key. [dict]
    :return: merged dictionaries. [dict]
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def nanogram_per_millit_to_nm(seq: str, ngml: float):
    """
    Helper function to transform units from ng/ml to nM.
    :param seq: aminoacid sequence. [str]
    :param ngml: measurement in ng/ml. [float]
    :return:
    """
    x = ProteinAnalysis(seq)
    return ((ngml * 10 ** -6) / x.molecular_weight()) / (1 * 10 ** -9)


# def setup_neptune_logger(config: dict[dict], neptune_api_token: list[str], tags=None):
def setup_neptune_logger_old(config: dict[dict], neptune_api_token: list[str], tags=None):
    """
    Setup neptune logger. Needs user unique api token from platfom.
    :param config: config dictionary containing all ymls configuations [dict[dict]]
    :param neptune_api_token: neptune ai api token. [str]
    :param tags: extra tags to add to the ones in config files. [list[str]]
    :return: Pytorch lightning wrapper for neptune logger. [pytorch_lightning.loggers.neptune.NeptuneLogger]
    """
    meta_tags = [config['model_name'],
                 config['dataset_params']['dataset_name'],
                 f"fold {config['dataset_params']['fold']}"]

    if tags is not None:
        meta_tags.extend(tags)

    # setup logger
    neptune_logger = NeptuneLogger(api_key=neptune_api_token,
                                   project=config['logger']['project_name'],
                                   name=config['model_name'],
                                   tags=meta_tags)
    neptune_logger.log_hyperparams({'dataset_params': config['dataset_params'],
                                           **config['train_params'],
                                           **config['model_params']})

    return neptune_logger

# def setup_neptune_logger_old(config: dict[dict], neptune_api_token: list[str], tags=None):
def setup_neptune_logger(config: dict[dict], neptune_api_token: list[str], tags=None):
    """
    Setup neptune logger. Needs user unique api token from platfom.
    :param config: config dictionary containing all ymls configuations [dict[dict]]
    :param neptune_api_token: neptune ai api token. [str]
    :param tags: extra tags to add to the ones in config files. [list[str]]
    :return: Pytorch lightning wrapper for neptune logger. [pytorch_lightning.loggers.neptune.NeptuneLogger]
    """
    meta_tags = [config['model_name'],
                 config['dataset_params']['dataset_name'],
                 f"fold {config['dataset_params']['fold']}"]

    if tags is not None:
        meta_tags.extend(tags)

    # setup logger
    neptune_logger = NeptuneLogger(api_key=neptune_api_token,
                                   project_name=config['logger']['project_name'],
                                   experiment_name=config['model_name'],
                                   params={'dataset_params': config['dataset_params'],
                                           **config['train_params'],
                                           **config['model_params']},
                                   tags=meta_tags)
    return neptune_logger

def setup_tensorboard_logger(config: dict[dict], args=None):
    """
    Setup tensorboard logger.
    :param config: config dictionary containing all ymls configuations [dict[dict]]
    :param neptune_api_token: neptune ai api token. [str]
    :param tags: extra tags to add to the ones in config files. [list[str]]
    :return: Pytorch lightning wrapper for neptune logger. [pytorch_lightning.loggers.neptune.NeptuneLogger]
    """

    # setup logger
    tensorboard_logger = TensorBoardLogger(save_dir='tensorboard',
                                    name=args.default_root_dir.split('/')[-1])
    
    tensorboard_logger.log_hyperparams(config)
    return tensorboard_logger

def output_to_nm(logscaled_output: float):
    """
    Transform BA network output to nM.
    :param logscaled_output: Network output. [float]
    :return: Binding affinity in nM. [float]
    """
    return np.exp((1 - logscaled_output) * np.log(50000))

def get_betabinomial_mean_variance(data: pd.DataFrame) -> np.array:
    """
    Calculate expected value, variance and regression target immunogenicity task (mean - variance),
    assuming betabinomial distribution for the experiment outcomes.
    :param data: immunogenicity dataframe. [pd.DataFrame]
    :return: expected value, variance and expected_value - variance. [np.array]
    """
    # get number of positive and negative
    trials = np.vstack([data.respond.values, (data.test - data.respond).values]).T

    # get mean and variance for betabinomial distribution with parametes from each set of peptide-hla experiments.
    stats = np.array([betabinom.stats(1, t[0] + 1, t[1] + 1) for t in trials])
    reg_target = stats[:, 0] - stats[:, 1]
    return np.vstack([stats[:, 0], stats[:, 1], reg_target])
