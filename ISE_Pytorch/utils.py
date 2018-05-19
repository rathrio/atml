import numpy
from torch.autograd import Variable
import torch.nn.init as I


def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    norm_c = input.norm(p, dim,keepdim = True)
    out = input / norm_c.clamp(min=eps).detach()
    return out.expand_as(input)


def xavier_weight(tensor):

    if isinstance(tensor, Variable):
        I.xavier_normal_(tensor.data)
        return tensor

    nin, nout = tensor.size()[0], tensor.size()[1]
    r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
    return tensor.normal_(0, r)


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            I.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)
    print('LSTM ortho init Done')
