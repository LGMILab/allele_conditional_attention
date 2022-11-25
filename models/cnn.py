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

class AffinityCNN(nn.Module):
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
                #  use_aa: bool,
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
        super(AffinityCNN, self).__init__()

        self.token_dim_hla = token_dim_hla
        self.hidden_dim_hla = hidden_dim_hla
        self.cnn_out_channels_hla = cnn_out_channels_hla
        self.use_assay_features = use_assay_features
        self.seq_len_hla = seq_len_hla
        self.num_classes = num_classes

        # self.use_esm = use_esm
        # self.use_aa = use_aa
        self.emb_type = emb_type
        # self.pool_type = pool_type

        # # pre-trained hla features projection
        # self.hla_embedding = nn.Linear(token_dim_hla, hidden_dim_hla)

        # # feature-wise convolution for HLA representation
        # self.hla_cnn = nn.Conv1d(in_channels=seq_len_hla, out_channels=cnn_out_channels_hla, kernel_size=1)


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

        # # peptide gru layer
        # self.gru_peptide = nn.GRU(token_dim_peptide,
        #                           hidden_dim_peptide,
        #                           n_layers_peptide,
        #                           dropout=dropout,
        #                           batch_first=True,
        #                           bidirectional=True)

        # Define Peptide,HLA CNN encoder
        hla_model = []

        hla_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        hla_model.append(nn.ReLU())
        hla_model.append(nn.MaxPool1d(kernel_size=3))
        hla_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        hla_model.append(nn.ReLU())
        hla_model.append(nn.MaxPool1d(kernel_size=3))
        hla_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        hla_model.append(nn.ReLU())
        hla_model.append(nn.MaxPool1d(kernel_size=3))
        hla_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        hla_model.append(nn.ReLU())
        hla_model.append(nn.MaxPool1d(kernel_size=3))

        peptide_model = []
        
        peptide_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        peptide_model.append(nn.ReLU())
        peptide_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))
        peptide_model.append(nn.ReLU())
        peptide_model.append(nn.MaxPool1d(kernel_size=3))
        peptide_model.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3))

        self.cnn_peptide = nn.Sequential(*peptide_model)
        self.cnn_hla = nn.Sequential(*hla_model)

        # fully connected layers for concatenated vector
        # concat_dim = (hidden_dim_peptide * 2 + hidden_dim_hla * cnn_out_channels_hla)
        concat_dim = hidden_dim_peptide + hidden_dim_hla
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

        out_hla = self.cnn_hla(hla.transpose(1,2)).squeeze()

        # peptide GRU
        out_peptide = self.cnn_peptide(peptide.transpose(1,2)).squeeze()

        # concatenate HLA and peptide features. takes output vector of last GRU unit.
        if self.use_assay_features:
            # out = torch.cat([out_hla[:, -1, :], out_peptide[:, -1, :], assay_features], dim=1)
            out = torch.cat([out_hla, out_peptide, assay_features], dim=1)
        else:
            out = torch.cat([out_hla, out_peptide], dim=1)

        # final fully connected layers
        pred = self.linear_block1(out)

        # out = self.linear_block1(out)
        # pred = self.linear_block2(out)

        # get embedding of the last layer first layer of linear_block2]
        if get_embeddings:
            return pred, list(self.linear_block2.children())[0](out)
        else:
            return pred

class ImmunogenicityGRU(nn.Module):
    def __init__(self,
                 token_dim_peptide: int,
                 token_dim_hla: int,
                 hidden_dim_peptide: int,
                 hidden_dim_hla: int,
                 seq_len_hla: int,
                 n_layers_peptide: int,
                 n_layers_proj_head: int,
                 cnn_out_channels_hla: int,
                 dropout: float,
                 use_process_features: bool,
                 process_features_dim: int,
                 num_classes: int=1):
        """
        Immunogenicity HLA-peptide GRU implementation with pre-trained representation for source sequence [HLA].
        Convolves feature-wise (opposed to token-wise) over pre-train representation to generate a feature vector
        for the whole sequence. [batch_size, seq_len, emb_dim] -> [batch_size, emb_dim].
        Does not use experiment related features (assay).
        :param token_dim_peptide: peptide sequence token (aminoacid) dimensionality. [int]
        :param token_dim_hla: hla sequence token (aminoacid) dimensionality. [int]
        :param hidden_dim_peptide: peptide sequence token hidden dimensionality. [int]
        :param hidden_dim_hla: hla sequence token hidden dimensionality. [int]
        :param seq_len_hla: HLA sequence length. [int] 
        :param n_layers_peptide: number of layers peptide GRU. [int]
        :param cnn_out_channels_hla: number of channels HLA pooling cnn. [int]
        :param dropout: GRU and final layers dropout. [float]
        """
        super(ImmunogenicityGRU, self).__init__()

        self.token_dim_hla = token_dim_hla
        self.hidden_dim_hla = hidden_dim_hla
        self.cnn_out_channels_hla = cnn_out_channels_hla
        self.seq_len_hla = seq_len_hla
        self.dropout = dropout
        self.use_process_features = use_process_features

        # pre-trained hla features projection
        self.hla_embedding = nn.Linear(token_dim_hla, hidden_dim_hla)

        # feature-wise convolution for HLA representation
        self.hla_cnn = nn.Conv1d(in_channels=seq_len_hla, out_channels=cnn_out_channels_hla, kernel_size=1)

        # peptide gru layer
        self.gru_peptide = nn.GRU(token_dim_peptide,
                                  hidden_dim_peptide,
                                  n_layers_peptide,
                                  dropout=dropout,
                                  batch_first=True,
                                  bidirectional=True)

        # fully connected layers for concatenated vector
        concat_dim = (hidden_dim_peptide * 2 + hidden_dim_hla * cnn_out_channels_hla)
        concat_dim = concat_dim + process_features_dim if use_process_features else concat_dim
        self.concat_dim = concat_dim

        linear_blocks = [nn.Linear(concat_dim, concat_dim // 2),
                                           nn.ReLU(),
                                           nn.Dropout(dropout),
                                           nn.InstanceNorm1d(concat_dim // 2) # for HRN Inst_norm
                                           ]
        for _ in range(n_layers_proj_head):
            linear_blocks.append(nn.Linear(concat_dim // 2, concat_dim // 2))
            linear_blocks.append(nn.ReLU())
            linear_blocks.append(nn.Dropout(dropout))
            linear_blocks.append(nn.InstanceNorm1d(concat_dim // 2)) # for HRN Inst_norm
            
        self.linear_block1 = nn.Sequential(*linear_blocks)

        self.linear_block2 = nn.Sequential(nn.Linear(concat_dim // 2, concat_dim // 4),
                                           nn.ReLU(),
                                        #    nn.Linear(concat_dim // 4, 1)
                                           nn.Linear(concat_dim // 4, num_classes)
                                           )
        
        self.ba_model = None
        self.stab_model = None
        self.el_model = None
        
    def add_ba_model(self, ba_model_params, ba_model=None):
        
        self.ba_model = ba_model
        
        ba_concat_dim = ba_model_params['cnn_pool_channels'] \
                    * ba_model_params['hidden_dim'] + 6 # 6 assay features dim
    
        # fully connected layers for concatenated vector
        self.concat_dim = self.concat_dim + ba_concat_dim//2
        
        self.linear_block1 = nn.Sequential(nn.Linear(self.concat_dim, self.concat_dim // 2),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        )
        self.linear_block2 = nn.Sequential(nn.Linear(self.concat_dim // 2, self.concat_dim // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.concat_dim // 4, 1)
                                        )
        
        # add assay features
        self.assay_features = torch.Tensor([[1., 0., 0., 1., 1., 0.]])
        
    def add_stab_model(self, stab_model_params, stab_model=None):
        
        self.stab_model = stab_model
        
        stab_concat_dim = stab_model_params['cnn_pool_channels'] * stab_model_params['hidden_dim']
    
        # fully connected layers for concatenated vector
        self.concat_dim = self.concat_dim + stab_concat_dim//2
        
        self.linear_block1 = nn.Sequential(nn.Linear(self.concat_dim, self.concat_dim // 2),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        )
        self.linear_block2 = nn.Sequential(nn.Linear(self.concat_dim // 2, self.concat_dim // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.concat_dim // 4, 1)
                                        )

    def add_el_model(self, el_model_params, el_model=None):
        
        # GRU based

        self.el_model = el_model
        
        el_concat_dim = (el_model_params['hidden_dim_peptide']*2 \
                        + el_model_params['hidden_dim_hla'] * el_model_params['cnn_out_channels_hla'])
    
        # fully connected layers for concatenated vector
        self.concat_dim = self.concat_dim + el_concat_dim//4
        
        self.linear_block1 = nn.Sequential(nn.Linear(self.concat_dim, self.concat_dim // 2),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        )
        self.linear_block2 = nn.Sequential(nn.Linear(self.concat_dim // 2, self.concat_dim // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.concat_dim // 4, 1)
                                        )

    def forward(self, peptide, hla, process_features=None, get_embeddings=False):

        # hla processing
        out_hla = self.hla_embedding(hla.reshape(-1, self.token_dim_hla))
        out_hla = out_hla.view(-1, self.seq_len_hla, self.hidden_dim_hla)
        out_hla = self.hla_cnn(out_hla).reshape(-1, self.cnn_out_channels_hla * self.hidden_dim_hla)

        # peptide GRU
        out_peptide, h = self.gru_peptide(peptide)

        # concatenate HLA and peptide features. takes output vector of last GRU unit.
        # out = torch.cat([out_hla, out_peptide[:, -1, :]], dim=1)
        if self.use_process_features:
            out = torch.cat([out_hla, out_peptide[:, -1, :], process_features], dim=1)
        else:
            out = torch.cat([out_hla, out_peptide[:, -1, :]], dim=1)
        
        # Add BA hidden vector from pretrained model
        if self.ba_model is not None:
            # ['purified MHC', 'competitive', 'fluorescence']
            assay_features = self.assay_features.type_as(peptide).tile((peptide.size(0), 1))

            _, ba_embed = self.ba_model(peptide, hla, assay_features=assay_features, get_embeddings=True)
            out = torch.cat([out, ba_embed], dim=1)
        else:
            ba_embed = None
            
        # Add Stability hidden vector from pretrained model
        if self.stab_model is not None:
            
            # _, stab_embed = self.stab_model(peptide, hla, get_embeddings=True)
            _, stab_embed, _, _ = self.stab_model(peptide, hla, get_embeddings=True) # for transformer setting
            out = torch.cat([out, stab_embed], dim=1)
        else:
            stab_embed = None

        # Add Eluted hidden vector from pretrained model
        if self.el_model is not None:
            
            # _, eluted_embed = self.eluted_model(peptide, hla, get_embeddings=True) # for GRU setting
            _, el_embed, _, _ = self.el_model(peptide, hla, get_embeddings=True) # for transformer setting
            out = torch.cat([out, el_embed], dim=1)
        else:
            el_embed = None

        # final fully connected layers
        out = self.linear_block1(out)
        pred = self.linear_block2(out)

        # get embedding of the last layer first layer of linear_block2]
        if get_embeddings:
            # return pred, list(self.linear_block2.children())[0](out)
            return pred, list(self.linear_block2.children())[0](out), ba_embed, stab_embed
        else:
            return pred
