# PyTorch NLI

## Instructions

**Install dependencies**

The following dependencies are required (versions used in brackets):
* Python (3.6.3)
* Pytorch (1.0.0)
* Numpy (1.15.4)
* Torchtext (for preprocessing) (0.4.0)
* SpaCy (for tokenization) (2.1.3)

For SpaCy you need to download the English model

```console
python -m spacy download en
```

**Download and prepare the datasets**

```console
./download_data.sh
```
This will download the needed datasets and word embeddings, including:
* [GloVe 840B 300D](https://nlp.stanford.edu/projects/glove/)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
* [SciTail](http://data.allenai.org/scitail/)
* [Breaking NLI](https://github.com/BIU-NLP/Breaking_NLI)

**Train a model**

Run the train_bilstm.sh script to train a BiLSTM model (InferSent) discussed in the lecture

```console
./train_bilstm.sh
```

Default settings for the SNLI dataset are as follows:

```console
python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type BiLSTMMaxPoolEncoder \
  --activation tanh \
  --optimizer sgd \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 512 \
  --hidden_dim 2048 \
  --layers 1 \
  --dropout 0 \
  --learning_rate 0.1 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --save_path results \
  --seed 1234
  ```

## References

[1] Aarne Talman, Anssi Yli-Jyrä and Jörg Tiedemann. 2018. [Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture](https://arxiv.org/pdf/1808.08762.pdf)

