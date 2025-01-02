import torch as t
from RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)

from model import  *
net = AMENet()
net.load_state_dict(t.load('/home/hjk/文档/论文代码/AMEN/three_network_wanzhen_uniformer_small_new_path_self_distillation_r1r4_d1d4_out1r1_out1d1_RGBD_SOD_2023_02_01_19_26_best.pth'))   ########gaiyixia

a = '/home/hjk/文档/RGBT-EvaluationTools/SalMap/'
b = 'AMENet'  ##########gaiyixia
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e


path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader1):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		out1 = net(image, depth)
		out = torch.sigmoid(out1[0])
		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()
		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')






