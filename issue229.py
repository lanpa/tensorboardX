# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

# import plot
# import session
# import session_mock
# import setting
# import util


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        input = input.view(len(input), 1, -1)
        output, self.hidden = self.lstm(input, self.hidden)
        output = self.fc(output)

        # combined = torch.cat((input, hidden), 1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)

        output = self.softmax(output)
        return output

    def init_hidden(self):
        # return torch.zeros(1, self.hidden_size)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


rnn = RNN(26, 3, 5)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.1)

# Keep track of losses for plotting
current_loss = 0
all_losses = []

writer = SummaryWriter()
one_input = None


def train_internal(category_tensor, vector_tensor):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    rnn.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    rnn.hidden = rnn.init_hidden()

    # Step 3. Run our forward pass.
    output = rnn(vector_tensor)
    global one_input
    if one_input is None:
        one_input = vector_tensor
    output = output.view(len(category_tensor), -1)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(output, category_tensor)
    loss.backward()

    optimizer.step()

    # Add parameters' gradients to their values, multiplied by learning rate
    # Model.parameters() is None while training:
    # https://discuss.pytorch.org/t/model-parameters-is-none-while-training/6830
    # for p in rnn.parameters():
    #     if p.grad is not None:
    #         p.data.add_(-setting.learning_rate, p.grad.data)

    return output, loss.item()


# Just return an output given a vector
def evaluate(vector_tensor):
    rnn.hidden = rnn.init_hidden()

    output = rnn(vector_tensor)

    return output


def predict(vector_tensor, n_predictions=2):
    with torch.no_grad():
        output = evaluate(vector_tensor)

        return setting.category_from_output(output)

        # # Get top N categories
        # topv, topi = output.topk(n_predictions, 1, True)
        # predictions = []
        #
        # for i in range(n_predictions):
        #     value = topv[0][i].item()
        #     category_index = topi[0][i].item()
        #     print('(%.2f) %s' % (value, all_categories[category_index]))
        #     predictions.append([value, all_categories[category_index]])
        # return predictions


def train_once(iter, n_iters, bot_count, vector, vector_tensor, category, category_tensor):
    global current_loss

    output, loss = train_internal(category_tensor, vector_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % setting.print_every == 0:
        guess, guess_i = setting.category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        if category == 'bot':
            bot_count += 1
            flag = 1
        else:
            flag = 0
        util.print_out('%d %d%% (%s) %.4f %s (count = %d) / %s %s: %d' % (
            iter, iter / n_iters * 100, time_since(start), loss, vector[0], len(vector), guess, correct, bot_count), flag)

    # Add current loss avg to list of losses
    writer.add_scalar('Loss', current_loss, iter)
    writer.add_scalars('Bot_Count', {'Total/10': (iter + 1) / 10, 'Bot': bot_count}, iter)
    if iter % setting.plot_every == 0:
        all_losses.append(current_loss / setting.plot_every)
        current_loss = 0

    return bot_count


def train():
    bot_count = 0
    for iter, n_iters, vector, vector_tensor, category, category_tensor in session.yield_matrix('data/data_old.csv'):
        bot_count = train_once(iter, n_iters, bot_count, vector, vector_tensor, category, category_tensor)

    plot.plot_loss(all_losses)


def test():
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(setting.n_categories, setting.n_categories)

    print('Test:')
    correct_count = 0
    tp = tn = 0.0
    fp = fn = 0.0
    # Go through a bunch of examples and record which are correctly guessed
    for iter, n_iters, vector, vector_tensor, category, category_tensor in session.yield_matrix('data/data_old.csv'):
        guess, guess_i = predict(vector_tensor)
        category_i = setting.all_categories.index(category)
        confusion[category_i][guess_i] += 1
        correct = '✓' if guess == category else '✗ (%s)' % category
        if guess == category:
            if guess == 'bot':
                tp += 1
            else:
                tn += 1
            correct_count += 1
            flag = 0
        else:
            if guess == 'bot':
                fp += 1
            else:
                fn += 1
            flag = 1
        accuracy = correct_count * 100.0 / (iter + 1)
        util.print_out('%s.. (count = %d) / %s %s: (%.1f)' % (vector[0], len(vector), guess, correct, accuracy), flag)

        # precision = 0 if tp + fp == 0 else tp * 100.0 / (tp + fp)
        # recall = 0 if tp + fn == 0 else tp * 100.0 / (tp + fn)
        # false_positive = 0 if fp + tn == 0 else fp * 100.0 / (fp + tn)
        # miss = 0 if tp + fn == 0 else fn * 100.0 / (tp + fn)
        # writer.add_scalars('Rate', {'Precision': precision, 'Recall': recall, 'False Positive': false_positive,
        # 'Miss': miss}, iter)
        writer.add_scalar('Accuracy', accuracy, iter)

        writer.add_scalars('Count', {'True Positive': tp, 'True Negative': tn, 'False Positive': fp, 'False Negative': fn}, iter)

        writer.add_scalars('Guess_Result', {'Guess': 1 if guess == 'bot' else 0, 'Ground Truth': 1 if category == 'bot' else 0}, iter)

    plot.plot_matrix(setting.all_categories, setting.n_categories, confusion)


if __name__ == '__main__':
    # input = util.vector_to_tensor([1, 3, 0.14764382294379175, 0.0, 204, 0.0518171270377934, 96, 63, 178, 41])
    # hidden =torch.zeros(1, n_hidden)
    #
    # output, next_hidden = rnn(input, hidden)
    # print(setting.category_from_output(output))

    # train()
    # test()

    # r = rnn(torch.autograd.Variable(torch.Tensor(2, setting.n_numbers), requires_grad=False))
    r = torch.autograd.Variable(torch.Tensor(2, 26), requires_grad=True)
    writer.add_graph(rnn, r, verbose=False)
