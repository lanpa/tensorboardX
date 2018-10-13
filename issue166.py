# generate dummy data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# write out the model
from tensorboardX import SummaryWriter

NUM_FEATURES = 2
SEQ_LEN = 8
BATCH_SIZE = 4

# prepare dataset
torch.manual_seed(5)
x = torch.ones([BATCH_SIZE, SEQ_LEN, NUM_FEATURES])
x = x.cumsum(dim=2)
L = torch.ones(4, dtype=torch.long)
for i in range(x.shape[0]):
    idx = 2*(i+1)
    if idx>=x.shape[1]:
        L[i]=x.shape[1]
    else:
        x[i, idx:, :] = 0
        L[i] = idx
    
y = torch.ones([4, 1], dtype=torch.long)
y[0:2] = 0

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_dim = 10
        self.input_size = NUM_FEATURES
        self.target_size = 2
        self.hidden = None
        self.bidirectional = False
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, 1,
        bidirectional=self.bidirectional)
        self.lstm.retain_variables = False

        self.hidden2target = nn.Linear(self.num_directions*self.hidden_dim, self.target_size)

    def init_hidden(self, batch_size):
        return (torch.zeros([self.num_directions, batch_size, self.hidden_dim], requires_grad=True, dtype=torch.float32), \
                torch.zeros([self.num_directions, batch_size, self.hidden_dim], requires_grad=True, dtype=torch.float32))

    def forward(self, seqs, T):
        # seqs is seq_len x batch_size x num_features
        # axes for input sequence:
        #   - The first axis is the sequence itself
        #   - the second indexes instances in the mini-batch
        #   - the third indexes elements of the input

        # reshape to match expected LSTM input
        # original: [batch, seq_len, input_feat]
        # after: [seq_len, batch, input_feat]
        #seqs = seqs.permute(1, 0, 2)

        # initialize hidden state
        self.hidden = self.init_hidden(seqs.size(0))
        
        # sort the batch by sequence length
        T, idx = T.sort(0, descending=True)
        seqs = seqs.index_select(0, idx)

        # pack the sequences
        seqs_packed = pack_padded_sequence(seqs, T.data, batch_first=True)

        lstm_out_packed, self.hidden = self.lstm(seqs_packed, self.hidden)

        # unpack the output
        lstm_out, _ = pad_packed_sequence(lstm_out_packed)

        # apply forward model to the final hidden state of seq - the [-1]
        targets = self.hidden2target(lstm_out[-1])

        return targets
    
model = LSTM()
model(x, L)

writer = SummaryWriter(comment='testing')
# below throws error
writer.add_graph(model, (x, L), verbose=False)
