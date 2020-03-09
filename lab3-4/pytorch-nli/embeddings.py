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
        prem = self.sentence_embedding(batch.premise)
        hypo = self.sentence_embedding(batch.hypothesis)
        answer = self.classifier(prem, hypo)
        return answer


class SentenceEmbedding(nn.Module):
    """
    Prepare and encode sentence embeddings with (Bi)LSTM encoder and max
    pooling
    """
    def __init__(self, config):
        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.embed_size, config.embed_dim)
        #self.word_embedding.weight.requires_grad=False
        self.encoder = eval(config.encoder_type)(config)

    def forward(self, input_sentence):
        sentence = self.word_embedding(input_sentence)
        sentence = Variable(sentence.data)
        embedding = self.encoder(sentence)
        return embedding

    def encode(self, input_sentence):
        embedding = self.encoder(input_sentence)
        return embedding


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM with max pooling
    """
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()
        self.config = config
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=0,
                           bidirectional=True)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
        embedding = self.rnn1(inputs, (h_0, c_0))[0]
        # Max pooling
        emb = self.max_pool(embedding.permute(1,2,0))
        emb = emb.squeeze(2)
        return emb


class LSTMEncoder(nn.Module):
    """
    Basic LSTM Encoder
    """
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=0,
                           bidirectional=False)
        self.batch_norm = nn.BatchNorm1d(config.hidden_dim)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
        embedding = self.rnn(inputs, (h_0, c_0))[1][0]
        embedding = embedding.squeeze(0)
        embedding = self.batch_norm(embedding)
        return embedding


class ConvEncoder(nn.Module):
    """
    Hierarchical Convolutional Encoder
    """
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        self.config = config
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.convnet1 = nn.Sequential(
            nn.Conv1d(config.embed_dim,
                      2*config.hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*config.hidden_dim,
                      2*config.hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*config.hidden_dim,
                      2*config.hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*config.hidden_dim,
                      2*config.hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        embedding = inputs

        embedding = embedding.transpose(0, 1).transpose(1, 2).contiguous()

        sentence = self.convnet1(embedding)
        emb1 = self.max_pool(sentence)

        sentence = self.convnet2(sentence)
        emb2 = self.max_pool(sentence)

        sentence = self.convnet3(sentence)
        emb3 = self.max_pool(sentence)

        sentence = self.convnet4(sentence)
        emb4 = self.max_pool(sentence)

        emb = torch.cat([emb1, emb2, emb3, emb4], 1)
        emb = emb.squeeze(2)
        return emb


class HBMP(nn.Module):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """
    def __init__(self, config):
        super(HBMP, self).__init__()
        self.config = config
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.cells = config.cells
        self.hidden_dim = config.hidden_dim
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=0,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=0,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.layers,
                            dropout=0,
                            bidirectional=True)


    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(1,2,0)).permute(2,0,1)

        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(1,2,0)).permute(2,0,1)

        out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(1,2,0)).permute(2,0,1)

        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(0)

        return emb

