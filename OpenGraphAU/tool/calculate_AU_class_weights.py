import numpy as np
import os

list_path_prefix = '/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/'

'''
example of content in 'AffWild2_train_label_fold1.txt':
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
'''


imgs_AUoccur = np.loadtxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'))
#imgs_AUoccur = imgs_AUoccur[:, [0,1,2,4,5,7,9,12,19,20,21,22]]
AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))

for i in range(imgs_AUoccur.shape[1]):
    AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0) / float(imgs_AUoccur.shape[0])

AU_weight = np.where(AUoccur_rate!=0, np.divide(1.0,AUoccur_rate), 0)
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_weight.txt'), AU_weight, fmt='%f', delimiter='\t')
# print(AU_weight.shape)