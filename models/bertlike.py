import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import LayerNorm
from models.transformer_base import BertlikeDecoderLayer, BertlikeDecoder
                          

""" attention pad mask """
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

""" attention decoder mask """
def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask

class PositionalEncoding(nn.Module):

    '''
    I think this should be modified, is this implementes w.r.t padded batched sequences?
    '''
    def __init__(self,
                 token_dim_seq: int,
                 max_len: int = 5000):
        """
        Positinal encoder as in vanilla transformer.
        :param token_dim_seq: Token hidden dimension. [int]
        :param max_len: Maximum sequence length. [int]
        """
        super(PositionalEncoding, self).__init__()

        positions = torch.arange(max_len).unsqueeze(1)
        feature_positions = torch.arange(token_dim_seq).unsqueeze(0)
        angles = positions / torch.pow(10000, (2 * feature_positions) / token_dim_seq)

        # apply sin to even indices in the array; 2i
        angles[:, 0::2] = torch.sin(angles[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer('pe', angles)

    def forward(self, x):
        pad = x.bool().float()

        x += self.pe[:x.size(1), :]
        # Given x is a batch of zero-padded sequence
        x *= pad 
        return x

class ProteinBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, 
                vocab_size=21,
                hidden_size=256,
                max_position_embeddings=14+182):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size)
        # self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AffinityBertlikePretrainedSource(nn.Module):
    def __init__(self,
                 token_dim_peptide: int,
                 token_dim_hla: int,
                 hidden_dim: int,
                 n_heads: int,
                 n_layers_decoder: int,
                 peptide_max_len: int,
                 hla_max_len: int,
                 cnn_pool_channels: int,
                 dropout: float,
                 activation: str,
                 use_assay_features: bool,
                 assay_features_dim: int,
                 num_classes: int,
                #  use_esm: bool,
                #  use_aa: bool,
                 emb_type: str,
                 pool_type: str):
        """
        Binding affinity HLA-peptide transformer implementation with pre-trained representation for source sequence [HLA].
        Applies CNN pooling to transformer generated token representations.
        Uses experiment related features as one hot vectors (assay).
        Rest of layers work as in vanilla implementation.
        :param token_dim_peptide: peptide input vector dimensionality. [int]
        :param token_dim_hla: hla input vector dimensionality. [int]
        :param hidden_dim: hidden representation dimensionality. [int]
        :param n_heads: decoder number of heads. [int]
        :param n_layers_decoder: decoder layers. [int]
        :param peptide_max_len: peptide sequence max length (for positional encoder). [int]
        :param cnn_pool_channels: trasnformer output tokens pooling channels. [int]
        :param dropout: decoder layer dropout. [float]
        :param activation: activation function of intermediate layers, ['relu', 'gelu']. [str]
        :param use_assay_features: use experiment related features. [bool]
        :param assay_features_dim: assay feature dimensionality. [int]
        """
        super(AffinityBertlikePretrainedSource, self).__init__()
        self.model_type = 'Bertlike'
        self.token_dim_peptide = token_dim_peptide
        self.cnn_pool_channels = cnn_pool_channels
        self.hidden_dim = hidden_dim
        self.token_dim_hla = token_dim_hla
        self.peptide_max_len = peptide_max_len
        # hla_max_len=182
        self.hla_max_len = hla_max_len
        self.use_assay_features = use_assay_features
        self.num_classes = num_classes
        self.n_layers_decoder = n_layers_decoder

        # self.use_esm = use_esm
        # self.use_aa = use_aa
        self.emb_type = emb_type
        self.pool_type = pool_type

        # if self.use_esm:
        if self.emb_type in ['aa+esm', 'esm2']:
            # hla features initial projection
            self.hla_embedding = nn.Linear(token_dim_hla, hidden_dim)

            # peptide features initial projection
            self.peptide_embedding = nn.Linear(token_dim_peptide, hidden_dim)
        
        
        # elif self.use_aa:
        if self.emb_type == 'aa2':
            # hla features initial projection
            self.hla_embedding = nn.Linear(token_dim_peptide, hidden_dim)

            # peptide features initial projection
            self.peptide_embedding = nn.Linear(token_dim_peptide, hidden_dim)
        
        if self.emb_type == 're':
            num_vocabs = 21 # Plus padding token
            self.hla_embedding = nn.Embedding(num_vocabs, hidden_dim)
            self.peptide_embedding = nn.Embedding(num_vocabs, hidden_dim)

        # # Old version
        # self.pos_encoder = PositionalEncoding(hidden_dim, max_len=peptide_max_len)

        # peptide positional encoder
        self.peptide_pos_encoder = PositionalEncoding(hidden_dim, max_len=peptide_max_len)
        self.hla_pos_encoder = PositionalEncoding(hidden_dim, max_len=hla_max_len) # for rand init embedding

        # HLA-peptide cross multihead attention. Source: HLA - Target: peptide
        decoder_layer = BertlikeDecoderLayer(hidden_dim, n_heads, hidden_dim, dropout, activation, batch_first=True)
            
        decoder_norm = nn.LayerNorm(hidden_dim)

        # self.transformer_peptide_decoder = TransformerDecoder(decoder_layer,
        # self.transformer_peptide_decoder = TransformerEncoder(decoder_layer,
        self.transformer_peptide_decoder = BertlikeDecoder(decoder_layer,
                                                            n_layers_decoder,
                                                            n_heads,
                                                            decoder_norm)
        
        # # Old version
        # # # feature-wise cnn for transformer decoder outputs
        # if self.pool_type == 'conv':
        #     self.cnn_pooling = nn.Conv1d(in_channels=peptide_max_len+hla_max_len, 
        #                              out_channels=cnn_pool_channels, kernel_size=1)
        #     concat_dim = cnn_pool_channels * self.hidden_dim
        # elif self.pool_type == 'average':
        #     concat_dim = self.hidden_dim
        
        # # feature-wise cnn for transformer decoder outputs
        if self.pool_type == 'conv':
            self.pooling = nn.Conv1d(in_channels=peptide_max_len+hla_max_len, 
                                     out_channels=cnn_pool_channels, kernel_size=1)
            concat_dim = cnn_pool_channels * self.hidden_dim
        elif self.pool_type == 'mlp':
            self.pooling = nn.Linear(in_features=peptide_max_len+hla_max_len, out_features=1)
            concat_dim = self.hidden_dim
        elif self.pool_type == 'average':
            concat_dim = self.hidden_dim
        # add ones-vector to represent [CLS] token for token pooling
        elif self.pool_type == 'token':
            self.cls_token = torch.ones(1,1,hidden_dim)
            concat_dim = self.hidden_dim
        
        # # fully connected layers for concatenated vector
        # concat_dim = cnn_pool_channels * self.hidden_dim if self.use_esm else self.hidden_dim
        concat_dim = concat_dim + assay_features_dim if self.use_assay_features else concat_dim

        # final fully connected layers
        '''
        model performs regression if num_classes given as 1.
        otherwise, this produces (num_classse - 1) dim 
        e.g. 4 classes -> 000, 001, 011, 111
        '''
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

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initiatialize parameters in the transformer model."
        """

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, peptide_tgt, hla_src, assay_features=None, 
                            get_embeddings=False, get_attention_weights=False):

        # # Old version
        # if self.use_esm:
        #     # HLA features projection (share weights across tokens)
        #     hla_src = self.hla_embedding(hla_src.view(-1, self.token_dim_hla))
        #     hla_src = hla_src.view(-1, 182, self.hidden_dim)

        #     # peptide embedding and positional encoding
        #     peptide_tgt = self.peptide_embedding(peptide_tgt.view(-1, self.token_dim_peptide))
        #     peptide_tgt = peptide_tgt.view(-1, self.peptide_max_len, self.hidden_dim)
        #     peptide_tgt = self.pos_encoder(peptide_tgt)
        
        # if self.use_esm:
        if self.emb_type in ['aa+esm', 'esm2']:
            # HLA features projection (share weights across tokens)
            hla_src = self.hla_embedding(hla_src.view(-1, self.token_dim_hla))
            hla_src = hla_src.view(-1, 182, self.hidden_dim)

            # peptide embedding and positional encoding
            peptide_tgt = self.peptide_embedding(peptide_tgt.view(-1, self.token_dim_peptide))
            peptide_tgt = peptide_tgt.view(-1, self.peptide_max_len, self.hidden_dim)
            peptide_tgt = self.peptide_pos_encoder(peptide_tgt)
                
        # elif self.use_aa:
        if self.emb_type == 'aa2':
            # hla embedding and positional encoding
            hla_src = self.hla_embedding(hla_src.view(-1, self.token_dim_peptide))
            hla_src = hla_src.view(-1, self.hla_max_len, self.hidden_dim)
            hla_src = self.hla_pos_encoder(hla_src)

            # peptide embedding and positional encoding
            peptide_tgt = self.peptide_embedding(peptide_tgt.view(-1, self.token_dim_peptide))
            peptide_tgt = peptide_tgt.view(-1, self.peptide_max_len, self.hidden_dim)
            peptide_tgt = self.peptide_pos_encoder(peptide_tgt)
            
        if self.emb_type == 're':
            hla_src = self.hla_embedding(hla_src)
            hla_src = self.hla_pos_encoder(hla_src)
            peptide_tgt = self.peptide_embedding(peptide_tgt)
            peptide_tgt = self.peptide_pos_encoder(peptide_tgt)

        # concat [CLS] token for peptide
        if self.pool_type == 'token':
            peptide_tgt = torch.cat([self.cls_token.expand(peptide_tgt.shape[0], -1, -1).type_as(peptide_tgt), 
                                    peptide_tgt], dim=1)

        # get attention weights for visulization if needed
        if get_attention_weights:
            # print('Not implemented Yet')
            # transformer encoder
            # output, saw, aw, saw_q, saw_k, aw_q, aw_k = self.transformer_peptide_decoder(peptide_tgt,
            #                                                                              hla_src,
            #                                                                              get_attention_weights=get_attention_weights)
            output, peptide_saw, hla_saw, peptide_saw_q, peptide_saw_k, hla_saw_q, hla_saw_k = \
                                                        self.transformer_peptide_decoder(peptide_tgt,
                                                                                         hla_src,
                                                                                         get_attention_weights=get_attention_weights)

            if self.pool_type == 'conv':
                # cnn pooling across output token representations
                output = self.pooling(output).reshape(-1, self.cnn_pool_channels * self.hidden_dim)
            elif self.pool_type == 'mlp':
                output = self.pooling(output.transpose(1,2)).reshape(-1, self.hidden_dim)
            elif self.pool_type == 'average':
                # average pool
                output = torch.mean(output, dim=1)
            elif self.pool_type == 'token':
                output = output[:,0,:].squeeze()

            # concatenate assay features if available
            output = torch.cat([output, assay_features], dim=1) if self.use_assay_features else output

            # last fully conected layers
            pred = self.linear_block1(output)

            # out_dict = {'saw': saw, 'aw': aw, 'saw_q': saw_q, 'saw_k': saw_k, 'aw_q': aw_q, 'aw_k': aw_k}
            
            out_dict = {'peptide_saw': peptide_saw,  'peptide_saw_q': peptide_saw_q, 'peptide_saw_k': peptide_saw_k, 
                        'hla_saw': hla_saw, 'hla_saw_q': hla_saw_q, 'hla_saw_k': hla_saw_k, 
                        }

            if get_embeddings:
                out_dict['embeds'] = list(self.linear_block1.children())[0](output)
            return pred, out_dict
            
        else:
            # transformer encoder
            output = self.transformer_peptide_decoder(peptide_tgt, hla_src, get_attention_weights=get_attention_weights)
            # output = self.transformer_peptide_decoder(torch.cat([peptide_tgt, hla_src], axis=1), 
            #                                           get_attention_weights=get_attention_weights)

            # # Old version
            # if self.pool_type == 'conv':
            #     # cnn pooling across output token representations
            #     output = self.cnn_pooling(output).reshape(-1, self.cnn_pool_channels * self.hidden_dim)
            # elif self.pool_type == 'average':
            #     # average pool
            #     output = torch.mean(output, dim=1)
            
            if self.pool_type == 'conv':
                # cnn pooling across output token representations
                output = self.pooling(output).reshape(-1, self.cnn_pool_channels * self.hidden_dim)
            elif self.pool_type == 'mlp':
                output = self.pooling(output.transpose(1,2)).reshape(-1, self.hidden_dim)
            elif self.pool_type == 'average':
                # average pool
                output = torch.mean(output, dim=1)
            elif self.pool_type == 'token':
                output = output[:,0,:].squeeze()

            # concatenate assay features if available
            output = torch.cat([output, assay_features], dim=1) if self.use_assay_features else output

            # last fully conected layers
            pred = self.linear_block1(output)

            # return embedding of the last layer first layer of linear_block2]
            if get_embeddings:
                return pred, list(self.linear_block1.children())[0](output)
            else:
                return pred