import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from classifier import FCClassifier


class NLIModel(nn.Module):
    """
    Main model class for the NLI task calling SentenceEmbedding and
    Classifier classes
    """
    def __init__(self, config):
        super(NLIModel, self).__init__()
        self.config = config
        self.sentence_embedding = SentenceEmbedding(config)
        self.classifier = FCClassifier(config)

    def forward(self, batch):
        prem = self.sentence_embedding(batch.premise) # Sentence embedding for the premise
        hypo = self.sentence_embedding(batch.hypothesis) # Sentence embedding for the hypothesis
        answer = self.classifier(prem, hypo)
        return answer


class SentenceEmbedding(nn.Module):
    """
    Prepare and encode sentence embeddings
    """
    def __init__(self, config):
        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.embed_size, config.embed_dim)
        #self.word_embedding.weight.requires_grad=False # Uncomment if you don't want to fine tune the word embeddings
        self.encoder = eval(config.encoder_type)(config)

    def forward(self, input_sentence):
        sentence = self.word_embedding(input_sentence)
        sentence = Variable(sentence.data)
        embedding = self.encoder(sentence) # Encode the sentence with the selected sentence encoder
        return embedding


class SumEncoder(nn.Module):
    """
    Basic Sum Encoder
    """
    def __init__(self, config):
        super(SumEncoder, self).__init__()
        self.config = config


    def forward(self, inputs):
        # Excervise 1: implement basic encoder summing the word vectors and taking the resulting 
        # vector as sentence embedding

        return emb


class LSTMEncoder(nn.Module):
    """
    Basic LSTM Encoder
    """
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config


    def forward(self, inputs):
        # Excervise 2: implement basic one-layer LSTM encoder taking the final hidden state as the 
        # sentence embedding

        return emb


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM with max pooling
    """
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()
        self.config = config


    def forward(self, inputs):
        # Excercise 3: implement bidirectional LSTM with max-pooling and take the max pooled output 
        # as the sentence embedding

        return emb


class ConvEncoder(nn.Module):
    """
    Hierarchical Convolutional Encoder
    """
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        self.config = config


    def forward(self, inputs): 
        # Excercise 4: implement convolutional 

        return emb

