import numpy as np
import numbers
import torchvision.transforms.functional as tvF
from torchvision import transforms
from PIL import Image

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch

class TCompose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask=None):
        if isinstance(img, list):
            img = [Image.fromarray(item.squeeze())
                     for item in img]
            self.PIL2Numpy = True
        # assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        # if self.PIL2Numpy:
        #     img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.uint8)

        return img, mask

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask=None):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            self.PIL2Numpy = True
        # assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.uint8)

        return img, mask

class ToTensor(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, input_data, gt_data=None):
        rdict = {}
        # input_data = sample['input']

        if isinstance(input_data, list):
            ret_input = [F.to_tensor(item)
                         for item in input_data]
        else:
            ret_input = F.to_tensor(input_data)

        if self.labeled:
            if gt_data is not None:
                if isinstance(gt_data, list):
                    ret_gt = [tvF.to_tensor(item)
                              for item in gt_data]
                else:
                    ret_gt = tvF.to_tensor(gt_data)
        else:
            ret_gt = gt_data
        return ret_input,ret_gt


class ToPIL(object):
    def __init__(self, labeled=True):
        self.labeled = labeled

    def sample_transform(self, sample_data):
        # Numpy array
        if not isinstance(sample_data, np.ndarray):
            input_data_npy = sample_data.numpy()
        else:
            input_data_npy = sample_data

        input_data_npy = np.transpose(input_data_npy, (1, 2, 0))
        input_data_npy = np.squeeze(input_data_npy, axis=2)
        input_data = Image.fromarray(input_data_npy)
        return input_data

    def __call__(self, input_data, gt_data=None):
        if isinstance(input_data, list):
            ret_input = [self.sample_transform(item)
                         for item in input_data]
        else:
            ret_input = self.sample_transform(input_data)

        if self.labeled:
            if isinstance(gt_data, list):
                ret_gt = [self.sample_transform(item)
                          for item in gt_data]
            else:
                ret_gt = self.sample_transform(gt_data)
        else:
            ret_gt = gt_data

        return ret_input,ret_gt


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input_data):
        input_data = tvF.normalize(input_data, self.mean, self.std)
        return input_data


class NormalizeInstance(object):
    """Normalize a tensor image with mean and standard deviation estimated
    from the sample itself.

    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __call__(self, input_data, gt_data=None):
        mean, std = input_data.mean(), input_data.std()
        input_data = tvF.normalize(input_data, [mean], [std])
        return input_data


class RandomRotation(object):
    def __init__(self, degrees, resample=False,expand=False, center=None,labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.labeled = labeled

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, input_data,gt_data=None):
        rdict = {}
        angle = self.get_params(self.degrees)
        input_data = tvF.rotate(input_data, angle,self.resample, self.expand,self.center)

        if self.labeled:
            gt_data = tvF.rotate(gt_data, angle,self.resample, self.expand,self.center)

        return input_data, gt_data

    
class RandomAffine(object):
    def __init__(self, degrees, translate=None,scale=None, shear=None,resample=False, fillcolor=0,labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.labeled = labeled

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def sample_augment(self, input_data, params):
        input_data = tvF.affine(input_data, *params, resample=self.resample,fillcolor=self.fillcolor)
        return input_data

    def label_augment(self, gt_data, params):
        gt_data = self.sample_augment(gt_data, params)
        np_gt_data = np.array(gt_data)
        np_gt_data = np.round(np_gt_data)
        np_gt_data[np_gt_data > np.array(gt_data).max()] = np.array(gt_data).max()
        gt_data = Image.fromarray(np_gt_data)
        return gt_data

    def __call__(self, input_data,gt_data=None):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        
        if isinstance(input_data, list):
            input_data_size = input_data[0].size
        else:
            input_data_size = input_data.size

        params = self.get_params(self.degrees, self.translate, self.scale,
                                 self.shear, input_data_size)

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params)
                         for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)

        if self.labeled:
            if isinstance(gt_data, list):
                ret_gt = [self.label_augment(item, params)
                          for item in gt_data]
            else:
                ret_gt = self.label_augment(gt_data, params)
        else:
            ret_gt = gt_data 
        return ret_input,ret_gt


class RandomTensorChannelShift(object):
    def __init__(self, shift_range):
        self.shift_range = shift_range

    @staticmethod
    def get_params(shift_range):
        sampled_value = np.random.uniform(shift_range[0],shift_range[1])
        return sampled_value

    def sample_augment(self, input_data, params):
        np_input_data = np.array(input_data,np.float32)
        np_input_data += params
        input_data = Image.fromarray(np.array(np_input_data,np.uint8))
        return input_data

    def __call__(self, input_data,gt_data=None):
        params = self.get_params(self.shift_range)

        if isinstance(input_data, list):
            ret_input = []
            ret_input.append(self.sample_augment(input_data[0], params))
            ret_input.append(input_data[1])
        else:
            ret_input = self.sample_augment(input_data, params)

        return ret_input,gt_data


class ElasticTransform(object):
    def __init__(self, alpha_range, sigma_range,p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params
        np_input_data = np.array(input_data)
        np_input_data = self.elastic_transform(np_input_data,param_alpha, param_sigma)
        input_data = Image.fromarray(np_input_data)
        return input_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)
        np_gt_data = self.elastic_transform(np_gt_data,param_alpha, param_sigma)
        np_gt_data = np.round(np_gt_data)
        np_gt_data[np_gt_data>np.array(gt_data).max()] = np.array(gt_data).max()
        gt_data = Image.fromarray(np_gt_data)
        return gt_data

    def __call__(self, input_data,gt_data=None):
        if np.random.random() < self.p:
            params = self.get_params(self.alpha_range,self.sigma_range)

            if isinstance(input_data, list):
                ret_input = [self.sample_augment(item, params)
                             for item in input_data]
            else:
                ret_input = self.sample_augment(input_data, params)

            if self.labeled:
                if isinstance(gt_data, list):
                    ret_gt = [self.label_augment(item, params)
                              for item in gt_data]
                else:
                    ret_gt = self.label_augment(gt_data, params)
            else:
                ret_gt = gt_data
        else:
            ret_input = input_data
            ret_gt = gt_data

        return ret_input,ret_gt


# TODO: Resample should keep state after changing state.
#       By changing pixel dimensions, we should be 
#       able to return later to the original space.

class AdditiveGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, input_data,gt_data=None):
        noise = np.random.normal(self.mean, self.std, input_data.size)
        noise = noise.astype(np.float32)
        np_input_data = np.array(input_data,np.float32)
        np_input_data += noise
        np_input_data = np.array(np_input_data,np.uint8)
        input_data = Image.fromarray(np_input_data)
        return input_data,gt_data


def mt_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: mt_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [mt_collate(samples) for samples in transposed]

    return batch

# if __name__ == '__main__':
#     import os
#     import torchvision as tv
#     savepath = "/home/jjc/Research/test/"
#     image1 = Image.open("/home/jjc/Research/ct_train_1010_image_124.pgm").convert("L")
#     image2 = Image.open("/home/jjc/Research/ct_train_1010_image_120.pgm").convert("L")
#     mask1 = Image.open("/home/jjc/Research/ct_train_1010_label_124.pgm").convert("L")
#     mask2 = Image.open("/home/jjc/Research/ct_train_1010_label_120.pgm").convert("L")

#     # image1 = np.array(Image.open("/home/jjc/Research/ct_train_1010_image_124.pgm").convert("L"))
#     # image2 = np.array(Image.open("/home/jjc/Research/ct_train_1010_image_120.pgm").convert("L"))
#     # mask1 = np.array(Image.open("/home/jjc/Research/ct_train_1010_label_124.pgm").convert("L"))
#     # mask2 = np.array(Image.open("/home/jjc/Research/ct_train_1010_label_120.pgm").convert("L"))
#     my_transform = Compose([
#         ElasticTransform(alpha_range=(28.0, 30.0),
#                                        sigma_range=(3.5, 4.0),
#                                        p=1, labeled=True),
#         RandomAffine(degrees=4.6,
#                    scale=(0.98, 1.02),
#                    translate=(0.03, 0.03),
#                    labeled=True),
#     ])
#     transform1 = ElasticTransform(alpha_range=(28.0, 30.0),
#                             sigma_range=(3.5, 4.0),
#                             p=0.3, labeled=False),
#     transform2 = RandomAffine(degrees=4.6,
#                             scale=(0.98, 1.02),
#                             translate=(0.03, 0.03),
#                             labeled=True),
    
#     imagedata = [image1,image2]
#     maskdata = [mask1,mask2]
#     if isinstance(imagedata, list):
#         print("list1")
#     if isinstance(maskdata, list):
#         print("list2")
#     # print(image1)
#     # print(mask1)
#     # my_transform = ElasticTransform(alpha_range=(28.0, 30.0),
#     #                         sigma_range=(3.5, 4.0),
#     #                         p=0.3, labeled=False)
#     image,mask = my_transform(imagedata,maskdata)
#     image1,image2 = image
#     mask1,mask2 =mask
#     image1.save(os.path.join(savepath,"img1-teacher_transform.pgm"))
#     image2.save(os.path.join(savepath,"img2-teacher_transform.pgm"))
#     mask1.save(os.path.join(savepath,"mask1-teacher_transform.pgm"))
#     mask2.save(os.path.join(savepath,"mask2-teacher_transform.pgm"))



