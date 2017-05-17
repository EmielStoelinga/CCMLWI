import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np





class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_cuda):
        super(RNN, self).__init__()
        self.use_cuda = use_cuda
        
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        #self.conv1 = nn.Conv1d(1, 250, kernel_size=5)
       # self.conv2 = nn.Conv1d(250, 40, kernel_size=5)
        self.softmax = nn.LogSoftmax()
        if self.use_cuda:
            self.cuda()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        #combined = combined.view(1, 1, combined.size()[1])
        #hidden_ = F.sigmoid(F.max_pool1d(self.conv1(combined), 2))
        #hidden_ = F.sigmoid(F.max_pool1d(self.conv2(hidden_), 2))
        #hidden_ = hidden_.view(1, -1)
        #hidden_ = torch.cat((hidden_, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        hidden = Variable(torch.zeros(1,self.hidden_size))
        if self.use_cuda:
           hidden = hidden.cuda()
        return hidden










class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_cuda):
        super(RNN1, self).__init__()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
        self.hidden_size = hidden_size
        #self.conv1 = nn.Conv2d(1, 250, kernel_size=5)
        #self.conv2 = nn.Conv2d(250, 40, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d(p=0)




        self.i2h = nn.Linear(input_size + self.hidden_size * self.hidden_size, self.hidden_size*self.hidden_size)
        self.i2o = nn.Linear(self.hidden_size*self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
      #  hidden = F.relu(F.max_pool2d(self.conv1(hidden), 2))
       # hidden = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(hidden)), 2))
        combined = torch.cat((input, hidden), 1)
        i2h = self.i2h(combined)
        output = self.i2o(i2h)
        output = self.softmax(output)
        return output, i2h

    def initHidden(self):
        hidden = Variable(torch.zeros(1,self.hidden_size * self.hidden_size))
        if self.use_cuda:
           hidden = hidden.cuda()
        return hidden
