import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class LSTMAutoEncoder(nn.Module):
    """
        Autoencoder based on an LSTM model with a single cell
        Args:
              input_length (int): input dimension
              code_length (int): hidden input representation (state dimension)
    """

    ##  Constructor
    def __init__(self, input_length, code_length):
        super(LSTMAutoEncoder, self).__init__()

        #  Attributes
        self.input_length = input_length
        self.code_length = code_length

        #  Layers
        self.encodeLayer = nn.LSTMCell(self.input_length, self.code_length)
        self.decodeLayer = nn.Linear(self.code_length, self.input_length)

    ##  Encode function
    def encode(self, x):
        # CODING (simple state encoding by the LSTMCell)
        hx, cx = self.encodeLayer(x)

        return cx

    ##  Decode function
    def decode(self, x):
        # DECODING (linear dense layer followed by an activation function [identity in this case, so none])
        x = self.decodeLayer(x)
        return x

    ##  Forward function
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


###    Execution    ###

##  Parameters

log_dir = '../tensorboard_logs/verdoliva_remote/LSTM_training_proto/demo_error'  # logging directory for tensorboard
train_batch_size = 5
input_length = 625
code_length = 100

if __name__ == '__main__':

    #  Model instantiation
    model = LSTMAutoEncoder(input_length, code_length)

    #  SummaryWriter instatiation
    writer = SummaryWriter()
    x = torch.randn(train_batch_size, input_length)
    writer.add_graph(model, x, verbose=True)
    writer.close()

