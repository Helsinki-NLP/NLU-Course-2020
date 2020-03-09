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


class BiLSTMMaxPoolEncoder(nn.Module):
    """
    Bidirectional LSTM with max pooling
    """
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
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


class BiLSTMInnerAttentionEncoder(nn.Module):
    """
    Bidirectional LSTM with inner attention
    """
    def __init__(self, config):
        super(BiLSTMInnerAttentionEncoder, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.gpu = config.gpu
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=config.layers,
                           dropout=0,
                           bidirectional=True)

        self.key_projection = nn.Linear(2*self.hidden_dim,
                                  2*self.hidden_dim,
                                  bias=False)
        self.proj_query = nn.Linear(2*self.hidden_dim,
                                   2*self.hidden_dim,
                                   bias=False)
        self.projection = nn.Linear(2*self.hidden_dim,
                                   2*self.hidden_dim,
                                   bias=False)
        self.query = nn.Embedding(2, 2*self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.hidden_dim).zero_())
        hidden_states, (last_h, last_c) = self.rnn1(inputs, (h_0, c_0))

        emb = self.attention(hidden_states, batch_size, temp=2)

        return emb


    def attention(self, hidden_states, batch_size, temp=2):

        output = hidden_states.transpose(0,1).contiguous()
        output_proj = self.projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = self.key_projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = torch.tanh(key)

        out = self.query(Variable(torch.LongTensor(batch_size*[0]).cuda(device=self.gpu))).unsqueeze(2)
        keys = key.bmm(out).squeeze(2) / temp
        keys = keys + ((keys == 0).float()*-1000)
        alphas = self.softmax(keys).unsqueeze(2).expand_as(key)
        atn_embed = torch.sum(alphas * output_proj, 1).squeeze(1)

        return atn_embed


class NGramEncoder(nn.Module):
    """
    Highly experimental stuff
    """
    def __init__(self, config):
        super(NGramEncoder, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.gpu = config.gpu
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=config.dropout,
                           bidirectional=True)
        self.key_projection = nn.Linear(2*config.hidden_dim,
                                  2*config.hidden_dim,
                                  bias=False)
        self.proj_query = nn.Linear(2*config.hidden_dim,
                                   2*config.hidden_dim,
                                   bias=False)
        self.projection = nn.Linear(2*config.hidden_dim,
                                   2*config.hidden_dim,
                                   bias=False)
        self.query = nn.Embedding(2, 2*config.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.LeakyReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        hidden = None
        for i in range(inputs.size()[0]):
            if inputs.size()[0] - i > 2:
                phrase = inputs[i:i+2, :, :]
            else:
                phrase = inputs[i:, :, :]

            if i == 0:
                h_0 = c_0 = Variable(phrase.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
            else:
                h_0 = ht1
                c_0 = ct1

            # Layer 1
            hidden_states, (ht1, ct1) = self.rnn1(phrase, (h_0, c_0))
            if i == 0:
                hidden = hidden_states
            else:
                hidden = torch.cat([hidden, hidden_states], 0)

        emb = self.max_pool(hidden.permute(1,2,0)).squeeze(2)

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

class HConvEncoder(nn.Module):
    """
    Hierarchical Convolutional Encoder
    """
    def __init__(self, config):
        super(HConvEncoder, self).__init__()
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

        sentence = self.convnet4(emb3)
        emb4 = self.max_pool(sentence)

        emb = torch.cat([emb2, emb3, emb4], 1)
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


class MASE(nn.Module):
    """
    Maximum Attention Sentence Encoder
    """
    def __init__(self, config):
        super(MASE, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.gpu = config.gpu
        #self.dropout = nn.Dropout(p=0.1)
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
        self.key_projection = nn.Linear(2*config.hidden_dim,
                                  2*config.hidden_dim,
                                  bias=False)
        self.proj_query = nn.Linear(2*config.hidden_dim,
                                   2*config.hidden_dim,
                                   bias=False)
        self.projection = nn.Linear(2*config.hidden_dim,
                                   2*config.hidden_dim,
                                   bias=False)
        self.query = nn.Embedding(2, 2*config.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.LeakyReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)


    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.hidden_dim).zero_())
        # Layer 1
        hidden_states1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        alphas1, emb1 = self.attention(hidden_states1, batch_size, temp=2)

        # Layer 2
        hidden_states2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        alphas2, emb2 = self.attention(hidden_states2, batch_size, temp=2)

        # Layer 3
        hidden_states3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        alphas3, emb3 = self.attention(hidden_states3, batch_size, temp=2)

        emb = torch.cat([emb1, emb2, emb3], 1)

        return emb


    def attention(self, hidden_states, batch_size, temp=2):

        output = hidden_states.transpose(0,1).contiguous()
        output_proj = self.projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        max_state = self.max_pool(hidden_states.permute(1,2,0)).permute(2,0,1)

        key = self.key_projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        max_proj = self.proj_query(max_state.squeeze(0)).unsqueeze(1).expand_as(key)

        key = self.activation(key+max_proj)
        out = self.query(Variable(torch.LongTensor(batch_size*[0]).cuda(device=self.gpu))).unsqueeze(2)
        keys = key.bmm(out).squeeze(2) / temp
        keys = keys + ((keys == 0).float()*-1000)
        # print(keys.size())
        # print(max_proj.size())
        alphas = self.softmax(keys).unsqueeze(2).expand_as(key)
        #alphas = self.dropout(alphas)
        atn_embed = torch.sum(alphas * output_proj, 1).squeeze(1)
        #embedding = torch.sum(alphas * max_proj, 1)
        embedding = torch.cat([atn_embed, max_state.squeeze(0)], 1)

        return alphas, embedding


class AttnEncoder(nn.Module):
    """
    Sentence Encoder based purely on word embeddings and inner attention - no recurrent layers
    """
    def __init__(self, config):
        super(AttnEncoder, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.gpu = config.gpu
        self.linear1 = nn.Linear(config.embed_dim,
                                config.hidden_dim)
        self.linear2 = nn.Linear(config.embed_dim,
                                 config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim,
                                   config.hidden_dim,
                                   bias=False)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.key_projection = nn.Linear(config.hidden_dim,
                                  config.hidden_dim,
                                  bias=False)
        self.proj_query = nn.Linear(config.hidden_dim,
                                   config.hidden_dim,
                                   bias=False)
        self.activation = nn.LeakyReLU()
        self.query = nn.Embedding(2, config.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        #self.dropout = nn.Dropout(p=0.3)weight.requires_grad=False

    def forward(self, inputs):
        batch_size = inputs.size()[1]

        hidden_states1 = self.linear1(inputs)
        atn_embed1 = self.attention(hidden_states1, batch_size, temp=2)
        #output_proj1 = self.dropout(output_proj1)

        hidden_states2 = self.linear2(inputs)
        atn_embed2 = self.attention(hidden_states2, batch_size, temp=3)
        #output_proj2 = self.dropout(output_proj2)

        embedding = torch.cat([atn_embed1, atn_embed2], 1)
        return embedding


    def attention(self, hidden_states, batch_size, temp=2):

        output = hidden_states.transpose(0,1).contiguous()
        output_proj = self.projection(output.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        max_state = self.max_pool(hidden_states.permute(1,2,0)).permute(2,0,1)

        key = self.key_projection(output.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        max_proj = self.proj_query(max_state.squeeze(0)).unsqueeze(1).expand_as(key)

        key = self.activation(key+max_proj)
        out = self.query(Variable(torch.LongTensor(batch_size*[0]).cuda(device=self.gpu))).unsqueeze(2)

        keys = key.bmm(out).squeeze(2) / temp
        keys = keys + ((keys == 0).float()*-1000)
        alphas = self.softmax(keys).unsqueeze(2).expand_as(key)
        #alphas = self.dropout(alphas)
        atn_embed = torch.sum(alphas * output_proj, 1).squeeze(1)

        #atn_embed = torch.sum(alphas * output_proj, 1)
        # embedding = torch.sum(alphas * max_proj, 1)
        embedding = torch.cat([atn_embed, max_state.squeeze(0)], 1)

        return embedding
