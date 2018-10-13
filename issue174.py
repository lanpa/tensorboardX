# https://github.com/pytorch/pytorch/blob/c82715ced5990b6fa3f35e15b9088a5d87adb700/torch/nn/_functions/rnn.py#L48

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

def visualize_graph(model, data, comment):
    with SummaryWriter(comment=comment) as writer:
        writer.add_graph(model, data, True)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        print(self.gru)
        print('==')
    def forward(self, word_inputs, hidden=None):
        """
        word_inputs: one batch of input data
                    (batch_size, seq_len) -> (64, 30)
        """
        input = word_inputs.transpose(0, 1).type(torch.LongTensor)  # Seqence first (30, 64)
        embedded = self.embedding((input))  # (30, 64, 100)
        return embedded
        # print(embedded.shape)
        # utput, hidden = self.gru(embedded)
        return output, hidden


encoder = EncoderRNN(37, 100)
sample_x = torch.rand(64,30).type(torch.IntTensor)

visualize_graph(encoder, sample_x, '_encoder')

