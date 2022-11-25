import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


def get_esm_representations(model, hla_df, hla_sequences, alphabet, batch_size=2, device='cuda'):
    """
    Get embeddings for a protein sequence with an ESM transformer model.
    Only works with the esm1b_t33_650M_UR50S model, which has 33 layers.
    To implement other models changing the hard-coded "33" values in the function.
    https://github.com/facebookresearch/esm
    :param model: esm model "esm1b_t33_650M_UR50S".
    :param hla_df:
    :param hla_sequences:
    :param alphabet:
    :param batch_size:
    :param device:
    :return:
    """
    # hla_seqs
    
    if '_' not in hla_df['hla'].iloc[0]: # only for MHC class I
        hlas = pd.merge(pd.DataFrame(hla_df.hla.unique(), columns=['hla']),
                        hla_sequences,
                        how='inner',
                        left_on='hla',
                        right_on='hla')
    else:
        hlas = hla_sequences

    # data
    hlas = [(a.hla, a.seq) for idx, a in hlas.iterrows()]

    #  data loader
    batch_converter = alphabet.get_batch_converter()
    batch_labels_hla, batch_strs_hla, batch_tokens_hla = batch_converter(hlas)
    data_loader = DataLoader(batch_tokens_hla, batch_size=batch_size)

    # inference
    # if not msa:
    results = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            output = model(data.to(device), repr_layers=[33], return_contacts=True)
        for k, v in output.items():
            if k == 'representations':
                output[k] = v[33].cpu()
            else:
                output[k] = v.cpu()
        results.append(output['representations'][:, 1:, :])
    results = torch.cat(results)
    hla_reps = {hla_name[0]: h.numpy() for hla_name, h in zip(hlas, results)}

    return hla_reps

def get_peptide_representations(model, peptide_df, alphabet, batch_size=2, device='cuda'):
    """
    Get embeddings for a protein sequence with an ESM transformer model.
    Only works with the esm1b_t33_650M_UR50S model, which has 33 layers.
    To implement other models changing the hard-coded "33" values in the function.
    https://github.com/facebookresearch/esm
    :param model: esm model "esm1b_t33_650M_UR50S".
    :param hla_df:
    :param hla_sequences:
    :param alphabet:
    :param batch_size:
    :param device:
    :return:
    """
    # # peptide_seqs
    # peptides = peptide_df.peptide.unique()

    # data
    peptides = [(a.peptide, a.peptide) for idx, a in peptide_df.iterrows()]

    #  data loader
    batch_converter = alphabet.get_batch_converter()
    batch_labels_peptide, batch_strs_peptide, batch_tokens_peptide = batch_converter(peptides)
    data_loader = DataLoader(batch_tokens_peptide, batch_size=batch_size)

    # inference
    
    results = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            output = model(data.to(device), repr_layers=[33], return_contacts=True)
        for k, v in output.items():
            if k == 'representations':
                output[k] = v[33].cpu()
            else:
                output[k] = v.cpu()
        results.append(output['representations'][:, 1:, :])
    results = torch.cat(results)
    peptide_reps = {peptide_name[0]: h.numpy() for peptide_name, h in zip(peptides, results)}
    
    return peptide_reps