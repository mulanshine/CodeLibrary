import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage import distance_transform_edt as distance

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, temperature=None, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.temperature = temperature
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(
            0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(
            1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(
            2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if self.temperature:
            target = target / self.temperature
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(
            n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average)
        return loss


def eightcorner_activation(x, size):
    """Retrieves neighboring pixels one the eight corners from a
    (2*size+1)x(2*size+1) patch.
    Args:
      x: A tensor of size [batch_size, height_in, width_in, channels]
      size: A number indicating the half size of a patch.
    Returns:
      A tensor of size [batch_size, height_in, width_in, channels, 8]
    """
    # Get the number of channels in the input.

    if x.dim() != 4:
        raise Exception('Only support for 4-D tensors!')

    b, c, h, w = x.size()
    # Pad at the margin.
    p = size
    x_pad = F.pad(x, [p]*4, mode='replicate')
    x_groups = []
    '''
    st_y    st_x    direction
      0       0      left-up
      0     size        up
      0    2*size    right-up

      ...

   '''
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue

            x_neighbor = x_pad[:, :, st_y:st_y+h, st_x:st_x+w]
            x_groups.append(x_neighbor)

    # output = [c.view(b,c,h,w,1) for c in x_groups]
    output = [torch.unsqueeze(c, -1) for c in x_groups]
    output = torch.cat(output, 4)
    return output


class LevelSetEnergyLoss(nn.Module):
    def __init__(self, size_average=True, alpha=0.001):
        super(LevelSetEnergyLoss, self).__init__()
        self.size_average = size_average
        self.alpha = alpha

    def forward(self, img, predict):
        softmax = F.softmax(predict, dim=1)
        n, c, h, w = softmax.size()
        loss = 0
        for i in range(c):
            class_i = softmax[:, i:i+1, :, :]
            mean = torch.sum(img * class_i) / (torch.sum(class_i)+1e-6)
            loss1 = torch.mean((img - mean)**2 * class_i)
            # class_i_pad = F.pad(class_i, [1,1,1,1], mode='replicate')
            # right
            # dx = class_i - class_i_pad[:,:, 1:1+h, 2:2+w]
            # dy = class_i - class_i_pad[:,:, 2:2+h, 1:1+w]
            dx = class_i[:, :, :, 0:w-1] - class_i[:, :, :, 1:w]
            dx = F.interpolate(
                dx, size=(h, w), mode='bilinear', align_corners=True)
            dy = class_i[:, :, 0:h-1, :] - class_i[:, :, 1:h, :]
            dy = F.interpolate(
                dy, size=(h, w), mode='bilinear', align_corners=True)
            loss2 = (dx + dy) / 2
            loss += (loss1 + self.alpha * loss2)
        return torch.mean(loss)


def eightway_activation(x):
    """Retrieves neighboring pixels/features on the eight corners from
    a 3x3 patch.
    Args:
    x: A tensor of size [batch_size, channels, height_in, width_in]
    Returns:
    A tensor of size [batch_size, channels, height_in, width_in, 8]
    """
    # Get the number of channels in the input.
    if x.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    b, c, h, w = x.size()
    # Pad at the margin.
    x = F.pad(x, [1]*4, mode='replicate')
    # Get eight neighboring pixels/features.
    x_groups = [
        x[:, :, 1:-1, :-2],  # left
        x[:, :, 1:-1, 2:],  # right
        x[:, :, :-2, 1:-1],  # up
        x[:, :, 2:, 1:-1],  # down
        x[:, :, :-2, :-2],  # left-up
        x[:, :, 2:, :-2],  # left-down
        x[:, :, :-2, 2:],  # right-up
        x[:, :, 2:, 2:]  # right-down
    ]

    output = [
        c.unsqueeze(-1) for c in x_groups
    ]
    # for c in output:
    #     print(c.shape)
    output = torch.cat(output, 4)
    return output


def eightway_affinity(x, size=1):
    neighbors = eightcorner_activation(x, size=size)
    affinity_groups = []
    for i in range(8):
        sim = x*neighbors[:, :, :, :, i]
        aff = torch.sum(sim, dim=1)
        affinity_groups.append(aff.unsqueeze(1))
    output = torch.cat(affinity_groups, dim=1)
    return output


def affinity_loss(probs):
    softmax = F.softmax(probs, dim=1)
    n, c, h, w = softmax.size()
    affinity = eightway_affinity(softmax, size=1)
    loss = 1.0 - affinity
    return loss.mean()


def margin_affinity_loss(probs, margin=0.2):
    softmax = F.softmax(probs, dim=1)
    n, c, h, w = softmax.size()
    affinity = eightway_affinity(softmax, size=1)
    loss = 1.0 - affinity
    loss = torch.where(loss > 1 - margin, loss, Variable(torch.FloatTensor(loss.data.size()).fill_(0)).cuda())
    return loss.mean()

def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t, axis=1):
    return simplex(t, axis) and sset(t, [0, 1])


class SurfaceLoss(nn.Module):
    def __init__(self,dim=None):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.dim = dim
        
    def __call__(self, probs, label):
        dist_maps = self.one_hot2dist(label)
        assert simplex(probs)
        assert not one_hot(dist_maps)
        pc = probs[:, 1, ...].type(torch.float32)
        dc = dist_maps[:, 1, ...].type(torch.float32)
        multipled = pc * dc
        loss = multipled.mean()
        return loss

    def one_hot2dist(self, label):
        assert one_hot(torch.Tensor(label), axis=0)
        C: int = len(label)

        dist = np.zeros_like(label)
        for c in range(C):
            posmask = label[c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                dist[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return dist
