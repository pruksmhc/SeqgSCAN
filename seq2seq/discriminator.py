import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

"""
Code from: https://github.com/suragnair/seqGAN
"""


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = torch.cuda.is_available()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input):
        # input dim         1000                                       # batch_size x seq_len
        # try:
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input).long()
        if self.gpu:
            input = input.cuda()
        # batch_size x seq_len x embedding_dim
        emb = self.embeddings(input)
        # except:
        #      import pdb; pdb.set_trace()

        out, _ = self.gru(emb)  # 4 x batch_size x hidden_dim
        out_last = out[:, -1,
              512:]  # Get the last layer output of the GRU, get the output of the last token in GRU (since it's autoregressive)
        out_last = torch.tanh(out_last)
        out_last = self.dropout_linear(out_last)
        out_last = self.hidden2out(out_last)  # batch_size x 1
        out_last = torch.sigmoid(out_last)
        return out_last

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len
        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)
