import torch as t
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import matplotlib
import numpy as np
import random
import torchvision





# gt 为png  其他两个为jpg格式
# Train NJU2K+LFSD
path_rgbt_RGB = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/rgb'
# path_rgbt_RGB = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/train/NJUNLPR/RGB'
lr = os.listdir(path_rgbt_RGB)
lr = [os.path.join(path_rgbt_RGB, img) for img in lr]
lr.sort()
# print(lr)

path_rgbt_GT = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/gt'
# path_rgbt_GT = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/train/NJUNLPR/GT'
gt = os.listdir(path_rgbt_GT)
gt = [os.path.join(path_rgbt_GT, gtimg) for gtimg in gt]
gt.sort()

path_rgbt_T = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/d'
# path_rgbt_T = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/train/NJUNLPR/depth'
depth = os.listdir(path_rgbt_T)
depth = [os.path.join(path_rgbt_T, dep) for dep in depth]
depth.sort()

path_rgbt_bound = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/boundary'
# path_rgbt_bound = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/train/NJUNLPR/boundary'
bound = os.listdir(path_rgbt_bound)
bound = [os.path.join(path_rgbt_bound, edge) for edge in bound]
bound.sort()

path_rgbt_background = '/media/hjk/shuju/轨道缺陷检测/Dataset/train/background'
background = os.listdir(path_rgbt_background)
background = [os.path.join(path_rgbt_background, edge) for edge in background]
background.sort()


# val NJU2K+LFSD
path_rgbt_val_rgb = '/media/hjk/shuju/轨道缺陷检测/Dataset/value/rgb'
# path_rgbt_val_rgb = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/val/NJUNLPR/RGB'
lrval = os.listdir(path_rgbt_val_rgb)
lrval = [os.path.join(path_rgbt_val_rgb, img) for img in lrval]
lrval.sort()

path_rgbt_val_GT = '/media/hjk/shuju/轨道缺陷检测/Dataset/value/gt'
# path_rgbt_val_GT = '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/val/NJUNLPR/GT'
gtval = os.listdir(path_rgbt_val_GT)
gtval = [os.path.join(path_rgbt_val_GT, gtimg) for gtimg in gtval]
gtval.sort()

path_rgbt_val_T= '/media/hjk/shuju/轨道缺陷检测/Dataset/value/d'
# path_rgbt_val_T= '/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/val/NJUNLPR/depth'
depthval = os.listdir(path_rgbt_val_T)
depthval = [os.path.join(path_rgbt_val_T, dep) for dep in depthval]
depthval.sort()



###Test
VT800_RGB = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/test/rgb')
VT800_RGB = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/test/rgb', img) for img in VT800_RGB]
VT800_RGB.sort()


VT800_GT = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/test/gt')
VT800_GT = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/test/gt', gtimg) for gtimg in VT800_GT]
VT800_GT.sort()


VT800_T = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/test/d')
VT800_T = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/test/d', dep) for dep in VT800_T]
VT800_T.sort()


# VT800_RGB = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/RGB')
# VT800_RGB = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/RGB', img) for img in VT800_RGB]
# VT800_RGB.sort()
#
#
# VT800_GT = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/GT')
# VT800_GT = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/GT', gtimg) for gtimg in VT800_GT]
# VT800_GT.sort()
#
#
# VT800_T = os.listdir('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/depth')
# VT800_T = [os.path.join('/media/hjk/shuju/轨道缺陷检测/Dataset/RGB-D SOD/test/STERE/depth', dep) for dep in VT800_T]
# VT800_T.sort()
############################################

# VT1000_RGB = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/RGB')
# VT1000_RGB = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/RGB', img) for img in VT1000_RGB]
# VT1000_RGB.sort()


# VT1000_GT = os.listdir('/home/hjk/文档/轨道缺陷检测/GT')
# VT1000_GT = [os.path.join('/home/hjk/文档/轨道缺陷检测/GT', gtimg) for gtimg in VT1000_GT]
# VT1000_GT.sort()


# VT1000_T = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/T')
# VT1000_T = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT1000/T', dep) for dep in VT1000_T]
# VT1000_T.sort()



############################################
# VT5000_RGB = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/RGB')
# VT5000_RGB = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/RGB', img) for img in VT5000_RGB]
# VT5000_RGB.sort()
#
#
# VT5000_GT = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/GT')
# VT5000_GT = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/GT', gtimg) for gtimg in VT5000_GT]
# VT5000_GT.sort()
#
#
# VT5000_T = os.listdir('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/T')
# VT5000_T = [os.path.join('/home/hjk/下载/Xiu Gai/Mobilnet 消融/RGBT_test/VT5000/T', dep) for dep in VT5000_T]
# VT5000_T.sort()



class NJUDateset(Dataset):
    def __init__(self, train, transform=None):
        self.train = train
        if self.train:
            self.lrimgs = lr
            self.depth = depth
            self.gt = gt
            self.bound = bound
            self.background = background


        else:
            self.lrimgs = lrval
            self.depth = depthval
            self.gt = gtval
            self.bound = bound

        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        depthPath = self.depth[index]
        gtPath = self.gt[index]
        gt_b = self.bound[index]
        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        depth = np.asarray(depth)
        gt = Image.open(gtPath)  # 0,255
        gt = np.asarray(gt).astype(np.float)
        if gt.max() == 255.:
            gt = gt / 255.
        gt_b = Image.open(gt_b)  # 0,255
        gt_b = np.asarray(gt_b).astype(np.float)
        if gt_b.max() == 255.:
            bound = gt_b / 255.

        sample = {'RGB': img, 'depth': depth, 'label': gt,'bound':bound}
        # sample = {'RGB': img, 'depth': depth, 'label': gt}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)


class TEST1(Dataset):
    def __init__(self, test, transform=None):
        self.test = test
        if self.test:
            self.lrimgs = VT800_RGB
            self.depth = VT800_T
            self.gt = VT800_GT


        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        depthPath = self.depth[index]
        gtPath = self.gt[index]
        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        depth = np.asarray(depth)
        gt = Image.open(gtPath)  # 0,255
        gt = np.asarray(gt).astype(np.float)
        # print(gt.shape)
        if gt.max() == 255.:
            gt = gt / 255.
        name = imgPath.split('/')[-1].split('.')[-2]
        sample = {'RGB': img, 'depth': depth, 'label': gt, 'name': name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)



#
# class TEST2(Dataset):
#     def __init__(self, test, transform=None):
#         self.test = test
#         if self.test:
#
#
#             self.lrimgs = VT1000_RGB
#             self.depth = VT1000_T
#             self.gt = VT1000_GT
#             # #
#
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         imgPath = self.lrimgs[index]
#         # imgboundpath = self.imgbound[index]
#         depthPath = self.depth[index]
#         gtPath = self.gt[index]
#         img = Image.open(imgPath)  # 0到255
#         img = np.asarray(img)
#         depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
#         depth = np.asarray(depth)
#         gt = Image.open(gtPath)  # 0,255
#         gt = np.asarray(gt).astype(np.float)
#         # print(gt.shape)
#         if gt.max() == 255.:
#             gt = gt / 255.
#         name = imgPath.split('/')[-1].split('.')[-2]
#         sample = {'RGB': img, 'depth': depth, 'label': gt, 'name': name}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#
#     def __len__(self):
#         return len(self.lrimgs)

# class TEST3(Dataset):
#     def __init__(self, test, transform=None):
#         self.test = test
#         if self.test:
#
#             self.lrimgs = VT5000_RGB
#             self.depth = VT5000_T
#             self.gt = VT5000_GT
#
#
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         imgPath = self.lrimgs[index]
#         # imgboundpath = self.imgbound[index]
#         depthPath = self.depth[index]
#         gtPath = self.gt[index]
#         img = Image.open(imgPath)  # 0到255
#         img = np.asarray(img)
#         depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
#         depth = np.asarray(depth)
#         gt = Image.open(gtPath)  # 0,255
#         gt = np.asarray(gt).astype(np.float)
#         # print(gt.shape)
#         if gt.max() == 255.:
#             gt = gt / 255.
#         name = imgPath.split('/')[-1].split('.')[-2]
#         sample = {'RGB': img, 'depth': depth, 'label': gt, 'name': name}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#
#     def __len__(self):
#         return len(self.lrimgs)

########################################################
# image_h = 224
# image_w = 224
image_h = 320
image_w = 320

class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label, bound = sample['RGB'], sample['depth'], sample['label'],sample['bound']
        # image, depth, label = sample['RGB'], sample['depth'], sample['label']

        # Bi-linear
        image = cv2.resize(image, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        bound = cv2.resize(bound, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        return {'RGB': image, 'depth': depth, 'label': label,'bound':bound}
        # return {'RGB': image, 'depth': depth, 'label': label}


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['RGB']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)


        return {'RGB': img_new, 'depth': sample['depth'], 'label': sample['label'], 'bound':sample['bound']}
        # return {'RGB': img_new, 'depth': sample['depth'], 'label': sample['label']}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label, bound = sample['RGB'], sample['depth'], sample['label'],sample['bound']
        # image, depth, label = sample['RGB'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()
            bound = np.fliplr(bound).copy()
        #
        return {'RGB': image, 'depth': depth, 'label': label,'bound':bound}
        # return {'RGB': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor


class Normalize(object):
    def __call__(self, sample):
        image, depth, label = sample['RGB'], sample['depth'], sample['label']
        image = image / 255.0
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # if depth.max() > 256.0:
        #     depth = depth / 31197.0
        # else:
        depth = depth / 255.0
        # depth = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(depth)
        label = label

        sample['RGB'] = image
        sample['depth'] = depth
        sample['label'] = label

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label, bound = sample['RGB'], sample['depth'], sample['label'],sample['bound']
        # image, depth, label = sample['RGB'], sample['depth'], sample['label']


        # bound1 = cv2.resize(bound, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)
        #
        # bound2 = cv2.resize(bound, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)
        #
        # bound3 = cv2.resize(bound, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)
        #
        # bound4 = cv2.resize(bound, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)
        #
        # bound5 = cv2.resize(bound, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)
        #


        # swap color axis because
        # numpy RGB: H x W x C
        # torch RGB: C X H X W
        image = image.transpose((2, 0, 1))

        # depth = image.transpose((0,1,2))

        # depth.squeeze()
        # depth = np.array([depth,depth,depth])
        depth = np.array([depth])
        depth = depth / 1.0
        # print(depth.shape)
        # depth = depth.transpose((2, 0, 1))

        label = np.expand_dims(label, 0).astype(np.float)
        # print(label.shape)
        bound = np.expand_dims(bound, 0).astype(np.float)
        # bound1 = np.expand_dims(bound1, 0).astype(np.float)
        # bound2 = np.expand_dims(bound2, 0).astype(np.float)
        # bound3 = np.expand_dims(bound3, 0).astype(np.float)
        # bound4 = np.expand_dims(bound4, 0).astype(np.float)
        # bound5 = np.expand_dims(bound5, 0).astype(np.float)
        # label2 = np.expand_dims(label2, 0).astype(np.float)
        # label3 = np.expand_dims(label3, 0).astype(np.float)
        # label4 = np.expand_dims(label4, 0).astype(np.float)
        # label5 = np.expand_dims(label5, 0).astype(np.float)
        # label6 = np.expand_dims(label6, 0).astype(np.float)
        return {'RGB': t.from_numpy(image).float(),
                'depth': t.from_numpy(depth).float(),
                'label': t.from_numpy(label).float(),
                # 'label2': t.from_numpy(label2).float(),
                # 'label3': t.from_numpy(label3).float(),
                # 'label4': t.from_numpy(label4).float(),
                # 'label5': t.from_numpy(label5).float(),
                # 'label6': t.from_numpy(label6).float(),
                'bound': t.from_numpy(bound).float(),
                # 'bound1': t.from_numpy(bound1).float(),
                # 'bound2': t.from_numpy(bound2).float(),
                # 'bound3': t.from_numpy(bound3).float(),
                # 'bound4': t.from_numpy(bound4).float(),
                # 'bound5': t.from_numpy(bound5).float(),
                }

class scaleNormtest(object):
    def __call__(self, sample):
        image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']

        # Bi-linear
        image = cv2.resize(image, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        name = name

        return {'RGB': image, 'depth': depth, 'label': label, 'name': name}



class Normalizetest(object):
    def __call__(self, sample):
        image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']
        image = image / 255.0
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # if depth.max() > 256.0:
        #     depth = depth / 31197.0
        # else:
        depth = depth / 255.0
        # depth = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(depth)

        label = label
        name = name
        sample['RGB'] = image
        sample['depth'] = depth
        sample['label'] = label
        sample['name'] = name
        return sample



class ToTensortest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']

        # Generate different label scales
        label2 = cv2.resize(label, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)

        label3 = cv2.resize(label, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)

        label4 = cv2.resize(label, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)

        label5 = cv2.resize(label, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)

        label6 = cv2.resize(label, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)



        # swap color axis because
        # numpy RGB: H x W x C
        # torch RGB: C X H X W
        image = image.transpose((2, 0, 1))
        # depth = depth.transpose((2, 0, 1))
        depth = np.array([depth])
        depth = depth / 1.0
        label = np.expand_dims(label, 0).astype(np.float)
        # label2 = np.expand_dims(label2, 0).astype(np.float)
        # label3 = np.expand_dims(label3, 0).astype(np.float)
        # label4 = np.expand_dims(label4, 0).astype(np.float)
        # label5 = np.expand_dims(label5, 0).astype(np.float)
        # label6 = np.expand_dims(label6, 0).astype(np.float)
        return {'RGB': t.from_numpy(image).float(),
                'depth': t.from_numpy(depth).float(),
                'label': t.from_numpy(label).float(),
                # 'label2': t.from_numpy(label2).float(),
                # 'label3': t.from_numpy(label3).float(),
                # 'label4': t.from_numpy(label4).float(),
                # 'label5': t.from_numpy(label5).float(),
                # 'label6': t.from_numpy(label6).float(),
                'name': name
                }


trainData = NJUDateset(train=True, transform=torchvision.transforms.Compose([
    scaleNorm(),
    RandomHSV((0.9, 1.1),
              (0.9, 1.1),
              (25, 25)),
    RandomFlip(),
    ToTensor(),
    Normalize()
]))

valData = NJUDateset(train=False, transform=torchvision.transforms.Compose([
    scaleNorm(),
    ToTensor(),
    Normalize(),
]
))


testData1 = TEST1(test=True, transform=torchvision.transforms.Compose([
    scaleNormtest(),
    ToTensortest(),
    Normalizetest(),
]
))

# testData2 = TEST2(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData3 = TEST3(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))

# testData4 = TEST4(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData5 = TEST5(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData6 = TEST6(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))
#
# testData7 = TEST7(test=True, transform=torchvision.transforms.Compose([
#     scaleNormtest(),
#     ToTensortest(),
#     Normalizetest(),
# ]
# ))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    sample = trainData[100]
    # name = sample['name']
    l1 = sample['label']
    # l2 = sample['label2']
    # l3 = sample['label3']
    # l4 = sample['label4']
    # l5 = sample['label5']
    # l6 = sample['label6']
    img = sample['RGB']
    depth = sample['depth']
    img1 = torchvision.transforms.ToPILImage()(img)

    plt.imshow(img1)
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2
    # plt.imshow('RGB', out_img)
    # cv2.imshow('rgb',img)
    # bound = sample['bound']
    # bound2 = sample['bound2']
    # bound3 = sample['bound3']
    # bound4 = sample['bound4']
    # bound5 = sample['bound5']
    # print(np.max(depth))
    # print(name)
    # print(l2.shape)
    # print(img.shape)
    # print(depth.shape)
    # print(depth)
    # print(bound.shape)
    # import numpy as np
    # uni1 = np.unique(l1)
    # print(uni1)
    # uni1 = np.unique(l2)
    # print(uni1)
    # uni1 = np.unique(l3)
    # print(uni1)
    # uni1 = np.unique(l4)
    # print(uni1)
    # uni1 = np.unique(l5)
    # print(uni1)
    # uni1 = np.unique(l6)
    # print(uni1)
    # bound = np.unique(bound)
    # print(bound)
    # bound2 = np.unique(bound2)
    # print(bound2)
    # bound3 = np.unique(bound3)
    # print(bound3)
    # bound4 = np.unique(bound4)
    # print(bound4)
    # bound5 = np.unique(bound5)
    # print(bound5)

# #

