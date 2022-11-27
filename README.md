
# Allele-conditional attention mechanism for HLA-peptide complex binding affinity prediction

This repository is the official implementation of [Allele-conditional attention mechanism for
HLA-peptide complex binding affinity prediction]. 

<img src="diagrams/figure1.png">
<img src="diagrams/figure2.png">

## Requirements

To install requirements:

```setup
conda env create -f environments.yml
```

Our dataset is available at https://drive.google.com/file/d/1ARVbJ1R1AElVHVo8D3A0jCwqUjRwiDXt/view?usp=sharing

Unzip data.tar file on ./ 

## Training

To train the model(s) in the paper, run commands like:

```train
bash run_5fold_train_{model_name}.sh
```
Available model_names are: 

{transformer(Conditional attention), 
 bertlike(Bert attention),
 cross_transformer(Cross attention),
 gru, cnn}

To evaluate certain model architecture, you can comment out rest of the code and run bash file.

You can select gpu device number for each fold with --gpu_id arguments.

Changing featurization scheme and pooling scheme is also possible,

by selecting --emb_type among {"aa2"(AA+AA), "aa+esm"(AA+ESM), "re"(Learned Embedding)} 

and --pool_type among {"average"(Mean pooling), "conv"(Learned weighting), "token"([CLS] token)}

## Evaluation

To evaluate model , run:

```eval
conda activate allele_conditional
bash run_5fold_val_{model_name}.sh
```
To evaluate certain model architecture, you can comment out rest of the code and run bash file.

You can select gpu device number for each fold with --gpu_id arguments.
