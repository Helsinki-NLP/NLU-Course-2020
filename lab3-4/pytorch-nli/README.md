# PyTorch NLI

## Instructions

**Install dependencies**

The following dependencies are required (versions used in brackets):
* Python (3.6 or higher)
* Pytorch (1.0.1)
* Numpy
* Torchtext (for preprocessing) (0.4.0)
* sacremoses (for tokenization)

**Download and prepare the datasets**

```console
./download_data.sh
```
This will download the needed datasets and word embeddings, including:
* [GloVe 840B 300D](https://nlp.stanford.edu/projects/glove/)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)

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
  --encoder_type BiLSTMEncoder \
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

