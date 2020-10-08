import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import argparse
import sys
sys.dont_write_bytecode = True

''' 
	
	This Network is designed for Few-Shot Learning Problem. 

'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer



def define_FewShotNet(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	FewShotNet = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model == 'Conv64F':
		FewShotNet = FourLayer_64F(norm_layer=norm_layer, **kwargs)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % which_model)
	init_weights(FewShotNet, init_type=init_type)

	if use_gpu:
		FewShotNet.cuda()

	if pretrained:
		FewShotNet.load_state_dict(model_root)

	return FewShotNet


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)



##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F 
# Input: One query image and a support set
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class FourLayer_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3, batch_size=4, shot_num=1):
		super(FourLayer_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                               # 64*21*21
		)
		
		self.classifier = Wass_CMS_Metric(neighbor_k=neighbor_k, batch_size=batch_size, shot_num=shot_num)  # 1*num_classes



	def forward(self, input1, input2):
		"""
			input1:    (300, 3, 84, 84)    ==> (batch_size * query_nums * way_nums, channels, H, W)
			input2:    (20, 3, 84, 84)     ==> (batch_size * shot_nums * way_nums, channels, H, W)
			batch_size:  4        
		"""

		# extract features of input1--query image
		q = self.features(input1).contiguous()  # (75*4) * 64 * 21 * 21
		q = q.view(q.size(0), q.size(1), -1)    # (75*4) * 64 * 441
		q = q.permute(0, 2, 1)                  # (75*4) * 441 * 64

		
		# extract features of input2--support set
		S = self.features(input2).contiguous()  # (25*4) * 64 * 21 *21
		S = S.view(S.size(0), S.size(1), -1)    # (25*4) * 64 * 441
		S = S.permute(0, 2, 1)                  # (25*4) * 441 * 64


		x, Q_S_List = self.classifier(q, S) # get Batch*num_classes

		return x, Q_S_List



#========================== Define an image-to-class layer ==========================#
class Wass_CMS_Metric(nn.Module):
	def __init__(self, neighbor_k=3, batch_size=4, shot_num=1):
		super(Wass_CMS_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.batch_size = batch_size
		self.shot_num = shot_num


	def cal_covariance_matrix_Batch(self, feature):   # feature: Batch * descriptor_num * 64
		n_local_descriptor = torch.tensor(feature.size(1)).cuda()
		feature_mean = torch.mean(feature, 1, True)   # Batch * 1 * 64
		feature = feature - feature_mean
		cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
		cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)

		return feature_mean, cov_matrix


	def support_remaining(self, S):   # S: 5 * 441 * 64

		S_new = []
		for ii in range(S.size(0)):
		
			indices = [j for j in range(S.size(0))]
			indices.pop(ii)
			indices = torch.tensor(indices).cuda()

			S_clone = S.clone()
			S_remain = torch.index_select(S_clone, 0, indices)           # 4 * 441 * 64
			S_remain = S_remain.contiguous().view(-1, S_remain.size(2))  # -1 * 64    
			S_new.append(S_remain.unsqueeze(0))

		S_new = torch.cat(S_new, 0)   # 5 * 1764 * 64 
		return S_new


	def wasserstein_distance_raw_Batch(self, mean1, cov1, mean2, cov2):
		'''
		   mean1: 75 * 1 * 64
		   cov1:  75 * 64 * 64
		   mean2: 5 * 1 * 64
		   cov2: 5 * 64 * 64
		'''
		mean_diff = mean1 - mean2.squeeze(1)                                        # 75 * 5 * 64
		cov_diff = cov1.unsqueeze(1) - cov2                                         # 75 * 5 * 64 * 64
		l2_norm_mean = torch.div(torch.norm(mean_diff, p=2, dim=2), mean1.size(2))  # 75 * 5 
		l2_norm_cova = torch.div(torch.norm(cov_diff, p=2, dim=(2,3)), mean1.size(2)*mean1.size(2))   # 75 * 5 

		return l2_norm_mean + l2_norm_cova



	# Calculate task-conditioned Wasserstein Distance
	def cal_wassersteinsimilarity(self, input1_batch, input2_batch):
		'''
			input1_batch: (75*4) * 441 * 64     75 query images
			input2_batch: (25*4) * 441 * 64     5-way 5-shot 
		'''
		Similarity_list = []
		Q_S_List = []
		input1_batch = input1_batch.contiguous().view(self.batch_size, -1, input1_batch.size(1), input1_batch.size(2))  # 4* 75 * 441 * 64
		input2_batch = input2_batch.contiguous().view(self.batch_size, -1, input2_batch.size(1), input2_batch.size(2))  # 4* 25 * 441 * 64


		for i in range(self.batch_size):

			input1 = input1_batch[i]
			input2 = input2_batch[i]

			# Calculate the mean and covariance of the all the query images
			query_mean, query_cov = self.cal_covariance_matrix_Batch(input1)   # query_mean: 75 * 1 * 64  query_cov: 75 * 64 * 64
			

			# Calculate the mean and covariance of the support set
			support_set = input2.contiguous().view(-1, 
				self.shot_num*input2.size(1), input2.size(2))                 # 5 * 441 * 64    
			s_mean, s_cov = self.cal_covariance_matrix_Batch(support_set)     # s_mean: 5 * 1 * 64  s_cov: 5 * 64 * 64


			# Find the remaining support set
			support_set_remain = self.support_remaining(support_set)
			s_remain_mean, s_remain_cov = self.cal_covariance_matrix_Batch(support_set_remain) # s_remain_mean: 5 * 1 * 64  s_remain_cov: 5 * 64 * 64


			# Calculate the Wasserstein Distance
			wasser_dis1 = self.wasserstein_distance_raw_Batch(query_mean, query_cov, s_mean, s_cov)
			wasser_dis2 = self.wasserstein_distance_raw_Batch(query_mean, query_cov, s_remain_mean, s_remain_cov)
			
			Similarity_list.append(-wasser_dis1 + wasser_dis2)


			# Store the mean and covariance
			parser = argparse.ArgumentParser()
			Q_S = parser.parse_args()
			Q_S.query_mean = query_mean
			Q_S.query_cov = query_cov
			Q_S.s_mean = s_mean
			Q_S.s_cov = s_cov
			Q_S.s_remain_mean = s_remain_mean
			Q_S.s_remain_cov = s_remain_cov
			Q_S_List.append(Q_S)


		return Similarity_list, Q_S_List


	def forward(self, x1, x2):

		Similarity_list, Q_S_List = self.cal_wassersteinsimilarity(x1, x2)

		return Similarity_list, Q_S_List

