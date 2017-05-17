import torch
import pandas
import random
from torch.autograd import Variable


class Data:

    def __init__(self, filename, use_cuda):
        self.whitelist = 'abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?'
        self.n_letters = len(self.whitelist)
        self.n_categories = 2
        self.readNews(filename)
        self.use_cuda = use_cuda


    def readNews(self, filename):
        df = pandas.read_csv(filename, sep='\t')
        self.headlines = df['normalized_headline'].as_matrix()
        self.stock = df['ms_today'].as_matrix()



    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            letter_index = self.whitelist.find(self.whitelist)
            tensor[li][0][letter_index] = 1
        return tensor

    def random_training_pair(self):
        chosen_category = random.choice([0,1])
        idx = random.choice(range(len(self.headlines)))
        category = self.stock[idx]
        while not category == chosen_category:
            idx = random.choice(range(len(self.headlines)))
            category = self.stock[idx]
        headline = self.headlines[idx]
        category_tensor = Variable(torch.LongTensor([category]))
        line_tensor = Variable(self.line_to_tensor(headline))
        if self.use_cuda:
            category_tensor = category_tensor.cuda()
            line_tensor = line_tensor.cuda()

        return category, headline, category_tensor, line_tensor
