import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import AffinityTransformerPretrainedSource

class FlankEncoding(nn.Module):
    def __init__(self,
                 token_dim_seq: int,
                 flank_types: int = 3):
        """
        Positinal encoder as in vanilla transformer.
        :param token_dim_seq: Token hidden dimension. [int]
        :param flank_types: Maximum sequence length. [int]
        """
        super(FlankEncoding, self).__init__()

        positions = torch.arange(flank_types).unsqueeze(1)
        feature_positions = torch.arange(token_dim_seq).unsqueeze(0)
        angles = positions / torch.pow(10000, (2 * feature_positions) / token_dim_seq)

        # apply sin to even indices in the array; 2i
        angles[:, 0::2] = torch.sin(angles[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer('fe', angles)
        
        self.len_n_flank = 4
        self.len_c_flank = 4

    def forward(self, x):
        """
        x: bs x max_seq_len x hidden_dim batch of padded tensor
        """

        pad = x.bool().float()

        n_fe = torch.zeros_like(x)
        center_fe = torch.zeros_like(x)
        n_fe[:,:4,:] = 1.
        n_fe = n_fe * self.fe[0]
        center_fe[:,4:,:] = 1.
        center_fe = center_fe * self.fe[1]
        
        c_fe = torch.cat([(1.-pad)[:,4:,:], torch.ones_like(x)[:,:4,:]], dim=1)
        c_fe = c_fe * (-self.fe[1]+self.fe[2])
        pep_fe = (n_fe + center_fe + c_fe)*pad
        x = x + pep_fe
        
        return x

class AffinityGRU(nn.Module):
    def __init__(self,
                 token_dim_peptide: int,
                 token_dim_hla: int,
                 hidden_dim_peptide: int,
                 hidden_dim_hla: int,
                 seq_len_hla: int,
                 n_layers_peptide: int,
                 cnn_out_channels_hla: int,
                 dropout: float,
                 use_assay_features: bool,
                 assay_features_dim: int,
                 num_classes: int,
                #  use_esm: bool,
                #  use_aa: bool
                 emb_type: str):
        """
        Binding affinity HLA-peptide GRU implementation with pre-trained representation for source sequence [HLA].
        Convolves feature-wise (opposed to token-wise) over pre-train representation to generate a feature vector
        for the whole sequence. [batch_size, seq_len, emb_dim] -> [batch_size, emb_dim].
        Uses experiment related features as one hot vectors (assay). Concatenated to GRU output.
        :param token_dim_peptide: peptide sequence token (aminoacid) dimensionality. [int]
        :param token_dim_hla: hla sequence token (aminoacid) dimensionality. [int]
        :param hidden_dim_peptide: peptide sequence token hidden dimensionality. [int]
        :param hidden_dim_hla: hla sequence token hidden dimensionality. [int]
        :param seq_len_hla: HLA sequence length. [int] 
        :param n_layers_peptide: number of layers peptide GRU. [int]
        :param cnn_out_channels_hla: number of channels HLA pooling cnn. [int]
        :param dropout: GRU and final layers dropout. [float]
        :param use_assay_features: use experiment related features. [bool]
        :param assay_features_dim: assay feature dimensionality. [int]
        
        :param num_classes: number of classes used . [int]
        """
        super(AffinityGRU, self).__init__()

        self.token_dim_hla = token_dim_hla
        self.hidden_dim_hla = hidden_dim_hla
        self.cnn_out_channels_hla = cnn_out_channels_hla
        self.use_assay_features = use_assay_features
        self.seq_len_hla = seq_len_hla
        self.num_classes = num_classes

        peptide_max_len=14
        self.peptide_max_len = peptide_max_len

        # self.use_esm = use_esm
        # self.use_aa = use_aa
        self.emb_type = emb_type
        # self.pool_type = pool_type

        # # pre-trained hla features projection
        # self.hla_embedding = nn.Linear(token_dim_hla, hidden_dim_hla)

        # if self.use_esm:
        if self.emb_type in ['aa+esm', 'esm2']:
            # hla features initial projection
            self.hla_embedding = nn.Linear(token_dim_hla, hidden_dim_hla)

            # peptide features initial projection
            self.peptide_embedding = nn.Linear(token_dim_peptide, hidden_dim_peptide)
        
        
        # elif self.use_aa:
        if self.emb_type == 'aa2':
            # hla features initial projection
            self.hla_embedding = nn.Linear(token_dim_peptide, hidden_dim_hla)

            # peptide features initial projection
            self.peptide_embedding = nn.Linear(token_dim_peptide, hidden_dim_peptide)
        
        if self.emb_type == 're':
            num_vocabs = 21 # Plus padding token
            self.hla_embedding = nn.Embedding(num_vocabs, hidden_dim_hla)
            self.peptide_embedding = nn.Embedding(num_vocabs, hidden_dim_peptide)

        # # Old version
        # self.pos_encoder = PositionalEncoding(hidden_dim, max_len=peptide_max_len)

        # # peptide positional encoder
        # self.peptide_pos_encoder = PositionalEncoding(hidden_dim, max_len=peptide_max_len)
        # self.hla_pos_encoder = PositionalEncoding(hidden_dim, max_len=hla_max_len) # for rand init embedding

        # # feature-wise convolution for HLA representation
        # self.hla_cnn = nn.Conv1d(in_channels=seq_len_hla, out_channels=cnn_out_channels_hla, kernel_size=1)

        # peptide gru layer
        self.gru_peptide = nn.GRU(hidden_dim_peptide,
                                  hidden_dim_peptide,
                                  n_layers_peptide,
                                  dropout=dropout,
                                  batch_first=True,
                                  bidirectional=True)

        self.gru_hla = nn.GRU(hidden_dim_hla,
                                  hidden_dim_hla,
                                  n_layers_peptide,
                                  dropout=dropout,
                                  batch_first=True,
                                  bidirectional=True)

        # fully connected layers for concatenated vector
        # concat_dim = (hidden_dim_peptide * 2 + hidden_dim_hla * cnn_out_channels_hla)
        concat_dim = (hidden_dim_peptide * 2) + (hidden_dim_hla * 2)
        concat_dim = concat_dim + assay_features_dim if self.use_assay_features else concat_dim

        if num_classes > 1:
            self.linear_block1 = nn.Sequential(nn.Linear(concat_dim, concat_dim // 2),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.BatchNorm1d(concat_dim // 2),
                                            nn.Linear(concat_dim // 2, num_classes-1))
        else:
            self.linear_block1 = nn.Sequential(nn.Linear(concat_dim, concat_dim // 2),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.BatchNorm1d(concat_dim // 2),
                                            nn.Linear(concat_dim // 2, 1))

    def forward(self, peptide, hla, assay_features=None, get_embeddings=False):

        hla = self.hla_embedding(hla)
        peptide = self.peptide_embedding(peptide)

        # # hla processing
        # out_hla = self.hla_embedding(hla.reshape(-1, self.token_dim_hla))
        # out_hla = out_hla.view(-1, self.seq_len_hla, self.hidden_dim_hla)
        # out_hla = self.hla_cnn(out_hla).reshape(-1, self.cnn_out_channels_hla * self.hidden_dim_hla)

        out_hla, _ = self.gru_hla(hla)

        # peptide GRU
        out_peptide, _ = self.gru_peptide(peptide)

        # concatenate HLA and peptide features. takes output vector of last GRU unit.
        if self.use_assay_features:
            out = torch.cat([out_hla[:, -1, :], out_peptide[:, -1, :], assay_features], dim=1)
        else:
            out = torch.cat([out_hla[:, -1, :], out_peptide[:, -1, :]], dim=1)

        # final fully connected layers
        pred = self.linear_block1(out)
        
        # out = self.linear_block1(out)
        # pred = self.linear_block2(out)

        # get embedding of the last layer first layer of linear_block2]
        if get_embeddings:
            return pred, list(self.linear_block2.children())[0](out)
        else:
            return pred
