import torch
from data import *
from model import *
import random
import time
import math
from tqdm import tqdm
from torch.cuda import is_available as cuda_available
import matplotlib.pyplot as plt
import torch.optim as optim

use_gpu = True
cuda = use_gpu and cuda_available()
filename = './data/combined_technology_news_stocks.csv'
from torch.cuda import is_available as cuda_available

data = Data(filename, cuda)
n_letters = data.n_letters
n_categories = data.n_categories
n_hidden = 144

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():                                                                                                               
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output, hidden
n_epochs = 100000
print_every = 1000
plot_every = 1000
rnn = RNN(n_letters, n_hidden, n_categories, cuda)
criterion = nn.NLLLoss()
learning_rate = 0.001
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    # Add parameters' gradients to their values, multiplied by learning rate

    return output, loss.data[0], hidden


if cuda:
    print 'Uses GPU'
# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in tqdm(range(1, n_epochs + 1)):
    category, line, category_tensor, line_tensor = data.random_training_pair()
    output, loss, hidden = train(category_tensor, line_tensor)
    current_loss += loss


    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        plt.clf()
        plt.matshow(hidden.data.cpu().numpy().reshape(12,12))
        plt.savefig('plt.jpg')
        guess = categoryFromOutput(output)
        correct = 'correct' if guess == category else 'false (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

        for i in range(50):
            category, line, category_tensor, line_tensor = data.random_training_pair()
            output, hidden = evaluate(line_tensor)
            print('line: {}, actual: {}, predicted: {}'.format(line, category, categoryFromOutput(output)))
            


    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

