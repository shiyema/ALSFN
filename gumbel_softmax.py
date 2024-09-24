import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps)).to(device)

def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, device, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot


def masked_softmax(logits, mask=None, dim=-1):
    eps = 1e-20
    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        ddif = len(probs.shape) - len(mask.shape)
        mask = mask.view(mask.shape + (1,)*ddif) if ddif>0 else mask
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def rao_gumbel(logits, device, temperature=1.0, mask=None, repeats=100, hard=True):
    logits_cpy = logits.detach()
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
    action = m.sample()
    action_bool = action.bool()

    logits_shape = logits.shape
    bs = logits_shape[0]

    E = torch.empty(logits_shape + (repeats,),
                    dtype=logits.dtype,
                    layout=logits.layout,
                    device=logits.device,
                    memory_format=torch.legacy_contiguous_format).exponential_()

    Ei = E[action_bool].view(logits_shape[:-1] + (repeats,))  # rv. for the sampled location

    wei = logits_cpy.exp()
    if mask is not None:
        wei = wei * mask.float() + 1e-20
    Z = wei.sum(dim=-1, keepdim=True)  # (bs, latdim, 1)
    EiZ = (Ei / Z).unsqueeze(-2)  # (bs, latdim, 1, repeats)

    new_logits = E / (wei.unsqueeze(-1))  # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()
    logits_diff = (new_logits - logits_cpy.unsqueeze(-1)).to(device)

    prob = masked_softmax((logits.unsqueeze(-1) + logits_diff) / temperature, mask, dim=-2).mean(dim=-1)
    action = action - prob.detach() + prob if hard else prob
    return action.view(logits_shape)

def st_gumbel_softmax(logits, device, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = (logits + gumbel_noise).to(device)
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y