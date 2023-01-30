from __future__ import unicode_literals, print_function, division
import pandas as pd
import os
import os
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


epochs = 1000000
plot_every = 1000
print_every = 5000
all_letters = string.ascii_lowercase + "."
n_letters = len(all_letters)
MIN_OCCURENCE = 3
max_len = 128


# Define Loss, Optimizer
m = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
learning_rate = 0.005

# TODO: Move out to util script
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


# TODO: Rename to reflect logic
def prepare_test_data(df):
    category_lines = (
        df.groupby(["last_name"])["state"]
        .apply(lambda grp: list(grp.value_counts().index))
        .to_dict()
    )
    all_names = list(category_lines.keys())
    return all_names, category_lines


def prepare_train_data(df):
    category_lines = (
        df.groupby(["state"])["last_name"]
        .apply(lambda grp: list(grp.value_counts().index))
        .to_dict()
    )
    all_categories = list(category_lines.keys())
    return category_lines, all_categories


def process_data(base_path):
    df = pd.read_csv(base_path)
    df = df[df.last_name != "LNU"]  # Remove last name unknows
    df = df[
        df.groupby("last_name")["last_name"].transform("count").ge(MIN_OCCURENCE)
    ]  # Remove all last names that occur less than MIN_OCCURANCE times
    return df


# Turning Names into Tensors
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a name into a <name_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def nameToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# Network Definition
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def model_init(n_categories, n_hidden):
    model = Model(n_letters, n_hidden, n_categories)
    hidden = model.initHidden()
    return model, hidden


# Training Data
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(X, y):
    category = randomChoice(y)
    line = randomChoice(X[category])
    category_tensor = torch.tensor([y.index(category)], dtype=torch.long)
    line_tensor = nameToTensor(line)
    return category, line, category_tensor, line_tensor


def randomTrainingExample_new(X, y):
    category = randomChoice(y)
    line = randomChoice(X[category])
    category_tensor = torch.tensor([y.index(category)], dtype=torch.long)
    line_tensor = nameToTensor(line)
    return category, line, category_tensor, line_tensor


def train(model, category_tensor, line_tensor, hidden, optimizer, clip=0.25):
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(m(output), category_tensor)
    # TODO: Penalize if the output is not from top-1
    loss.backward()
    clip = 0.25
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return output, loss.item()


# Just return an output given a line
def evaluate(model, line_tensor):
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output


# Predict
def predict_standalone(model, x, y, n_predictions=3):
    # x should be caseted to .lower()
    with torch.no_grad():
        output = evaluate(model, nameToTensor(x))
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, sorted=True)
        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append(y[category_index])
    return predictions


def split_tr_te(df):
    samplelist = df["last_name"].unique()
    training_samp, test_samp = train_test_split(
        samplelist, train_size=0.8, test_size=0.2, random_state=5, shuffle=True
    )
    training_data = df[df["last_name"].isin(training_samp)]
    test_data = df[df["last_name"].isin(test_samp)]
    return training_data, test_data


if __name__ == "__main__":
    base_dir = "path/to/instate_data"
    fid = "instate_processed.csv.gz"
    # Data
    print("Processing data")
    df = process_data(os.path.join(base_dir, fid))
    training_data, test_data = split_tr_te(df)
    X_tr, y_tr = prepare_train_data(training_data)
    X_te, y_te = prepare_test_data(test_data)
    n_categories = len(y_tr)
    # Model Init
    _model, hidden = model_init(n_categories, max_len)
    optimizer = torch.optim.SGD(_model.parameters(), lr=learning_rate)

    # Training
    current_loss = 0
    all_losses = []

    start = time.time()
    for iter in range(1, epochs + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(X_tr, y_tr)
        output, loss = train(_model, category_tensor, line_tensor, hidden, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            print(
                "%d %d%% (%s) %.4f "
                % (
                    iter,
                    iter / epochs * 100,
                    timeSince(start),
                    loss,
                )
            )

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Export model
    torch.save(_model.state_dict(), os.path.join(base_dir, "instate.pt"))
    # plot Losses
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    # # Eval
    top_3 = []
    for _x in X_te:
        y_gt = y_te[_x][0]
        y_pred = predict_standalone(_model, _x, y_tr)
        if y_gt in y_pred:
            top_3.append(1)
        else:
            top_3.append(0)
    print(f"Top-3 Accuracy: {sum(top_3)/ len(top_3)*100}")
