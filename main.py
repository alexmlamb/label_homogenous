'''
Simple MNIST classifier: test the effect of training with homogenous batches (same label in batch) vs. random batches.  
'''

import torch
from torch.autograd import Variable, grad
from torch import optim
import gzip
import cPickle as pickle


def init_params():

    params = {}

    params['W1'] = Variable(0.01 * torch.randn(784,512), requires_grad=True).cuda()
    params['W2'] = Variable(0.01 * torch.randn(512,10), requires_grad=True).cuda()

    return params

def network(p, x, ytrue):

    x = Variable(torch.from_numpy(x)).cuda()
    ytrue = Variable(torch.from_numpy(ytrue).type(torch.LongTensor)).cuda()

    sm = torch.nn.Softmax()
    relu = torch.nn.ReLU()
    nll = torch.nn.functional.nll_loss

    h1 = relu(torch.matmul(x, p['W1']))

    y = sm(torch.matmul(h1, p['W2']))

    loss = nll(y, ytrue).sum()

    return loss

if __name__ == "__main__":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid
    testx, testy = test

    x = trainx[0:64]
    y = trainy[0:64]

    print x.shape

    p = init_params()

    loss = network(p,x,y)

    print loss



