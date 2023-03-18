import os
import numpy as np
import pandas as pd


print("processing AFFW-2------------------------------------------------------------")

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']


list_path_prefix = '/work/vita/datasets/Aff-Wild2/Third_ABAW_annotations/AU_Detection_Challenge/'

train_path = 'Train_Set'
val_path = 'Validation_Set'
test_path = 'Validation_Set'

label_root = '/work/vita/datasets/Aff-Wild2/Third_ABAW_annotations/AU_Detection_Challenge/'

train_list = os.listdir(os.path.join(label_root, train_path))

train_labels = os.path.join(list_path_prefix, 'AFFW-2_train_label.txt')
with open(train_labels, 'w') as  f:
    i = 0

val_list = os.listdir(os.path.join(label_root, val_path))

val_labels = os.path.join(list_path_prefix, 'AFFW-2_val_label.txt')
with open(val_labels, 'w') as  f:
    i = 0

test_list = os.listdir(os.path.join(label_root, test_path))

test_labels = os.path.join(list_path_prefix, 'AFFW-2_test_label.txt')
with open(test_labels, 'w') as  f:
    i = 0


train_img_path = os.path.join(list_path_prefix, 'AFFW-2_train_img_path.txt')
with open(train_img_path, 'w') as f:
    i = 0
val_img_path = os.path.join(list_path_prefix, 'AFFW-2_val_img_path.txt')
with open(val_img_path, 'w') as f:
    i = 0
test_img_path = os.path.join(list_path_prefix, 'AFFW-2_test_img_path.txt')
with open(test_img_path, 'w') as f:
    i = 0




au_labels = []
au_img_path = []
for train_txt in train_list:
    with open(os.path.join(os.path.join(label_root, train_path), train_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(train_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_train_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_train_image_label[:, index] = au_labels[:, i]

with open(train_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_train_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(train_labels, AFFW_train_image_label ,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(AFFW_train_image_label)




au_labels = []
au_img_path = []
for val_txt in val_list:
    with open(os.path.join(os.path.join(label_root, val_path), val_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(val_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_val_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_val_image_label[:, index] = au_labels[:, i]

with open(val_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_val_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(val_labels, AFFW_val_image_label ,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(AFFW_val_image_label)





au_labels = []
au_img_path = []
for test_txt in test_list:
    with open(os.path.join(os.path.join(label_root, test_path), test_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(test_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_test_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_test_image_label[:, index] = au_labels[:, i]

with open(test_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_test_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(test_labels, AFFW_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(AFFW_test_image_label)


#
# print(len(new_dataset_train_img_list))
# print(len(new_dataset_val_img_list))
# print(len(new_dataset_test_img_list))

new_dataset_train_label_list = np.concatenate(new_dataset_train_label_list, axis=0)
new_dataset_val_label_list = np.concatenate(new_dataset_val_label_list, axis=0)
new_dataset_test_label_list = np.concatenate(new_dataset_test_label_list, axis=0)


sub_list = [0,1,2,4,7,8,11]

for i in range(new_dataset_train_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_train_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_train_label_list[i, main_au_index] = 1


for i in range(new_dataset_val_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_val_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_val_label_list[i, main_au_index] = 1

for i in range(new_dataset_test_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_test_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_test_label_list[i, main_au_index] = 1

np.savetxt('ME-GraphAU/data/AffWild2/list/AffWild2_train_label.txt', new_dataset_train_label_list ,fmt='%d', delimiter=' ')
np.savetxt('ME-GraphAU/data/AffWild2/list/AffWild2_val_label.txt', new_dataset_val_label_list ,fmt='%d', delimiter=' ')
np.savetxt('ME-GraphAU/data/AffWild2/list/AffWild2_test_label.txt', new_dataset_test_label_list ,fmt='%d', delimiter=' ')

with open('ME-GraphAU/data/AffWild2/list/AffWild2_train_img_path.txt', 'w+') as f:
    for line in new_dataset_train_img_list:
        f.write(line)

with open('ME-GraphAU/data/AffWild2/list/AffWild2_val_img_path.txt', 'w+') as f:
    for line in new_dataset_val_img_list:
        f.write(line)

with open('ME-GraphAU/data/AffWild2/list/AffWild2_test_img_path.txt', 'w+') as f:
    for line in new_dataset_test_img_list:
        f.write(line)

# print(new_dataset_train_label_list.shape)
# print(new_dataset_val_label_list.shape)
# print(new_dataset_test_label_list.shape)

# new_dataset_train_label_list[new_dataset_train_label_list==-1] = 0
# new_dataset_val_label_list[new_dataset_val_label_list==-1] = 0
# new_dataset_test_label_list[new_dataset_test_label_list==-1] = 0
#
# print(new_dataset_train_label_list.sum(0))
# print(new_dataset_val_label_list.sum(0))
# print(new_dataset_test_label_list.sum(0))






# AffWild2_Sequence_split = [['F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'],
# 					   ['M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005'],
# 					   ['F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014']]

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

# tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
# label_folder = '../data/AffWild2/AUCoding/AU_OCC'
# list_path_prefix = '../data/AffWild2/list/'

# def get_AUlabels(seq, task, path):
# 	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
# 	usecols = ['0', '1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
# 	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
# 	frames = [str(item) for item in list(df.index.values)]
# 	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
# 	labels = df.values
# 	返回的frames是list，值是排好序的int变量，指示对应的帧。labels是N*12的np.ndarray，对应AU标签
# 	return frames_path, labels

# ##################################################################################################
# with open(list_path_prefix + 'AffWild2_test_img_path_fold3.txt','w') as f:
#     u = 0

# sequences = AffWild2_Sequence_split[0]
# frames = None
# labels = None
# len_f = 0
# for seq in sequences:
# 	for t in tasks:
# 		temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
# 		if frames is None:
# 			labels = temp_labels
# 			frames = temp_frames  # str list
# 		else:
# 			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
# 			frames = frames + temp_frames  # str list
# AffWild2_image_path_list_part1 = frames
# AffWild2_image_label_part1 = labels

# for frame in AffWild2_image_path_list_part1:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_test_img_path_fold3.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# np.savetxt(list_path_prefix + 'AffWild2_test_label_fold3.txt', AffWild2_image_label_part1 ,fmt='%d', delimiter=' ')


# ##################################################################################################
# with open(list_path_prefix + 'AffWild2_test_img_path_fold2.txt','w') as f:
#     u = 0

# sequences = AffWild2_Sequence_split[1]
# frames = None
# labels = None
# len_f = 0
# for seq in sequences:
# 	for t in tasks:
# 		temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
# 		if frames is None:
# 			labels = temp_labels
# 			frames = temp_frames  # str list
# 		else:
# 			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
# 			frames = frames + temp_frames  # str list
# AffWild2_image_path_list_part2 = frames
# AffWild2_image_label_part2 = labels

# for frame in AffWild2_image_path_list_part2:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_test_img_path_fold2.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# np.savetxt(list_path_prefix + 'AffWild2_test_label_fold2.txt', AffWild2_image_label_part2, fmt='%d', delimiter=' ')

# ##################################################################################################
# with open(list_path_prefix + 'AffWild2_test_img_path_fold1.txt','w') as f:
#     u = 0

# sequences = AffWild2_Sequence_split[2]
# frames = None
# labels = None
# len_f = 0
# for seq in sequences:
# 	for t in tasks:
# 		temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
# 		if frames is None:
# 			labels = temp_labels
# 			frames = temp_frames  # str list
# 		else:
# 			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
# 			frames = frames + temp_frames  # str list
# AffWild2_image_path_list_part3 = frames
# AffWild2_image_label_part3 = labels

# for frame in AffWild2_image_path_list_part3:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_test_img_path_fold1.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# np.savetxt(list_path_prefix + 'AffWild2_test_label_fold1.txt', AffWild2_image_label_part3, fmt='%d', delimiter=' ')


# ###############################################################################
# with open(list_path_prefix + 'AffWild2_train_img_path_fold1.txt','w') as f:
#     u = 0
# train_img_label_fold1_list = AffWild2_image_path_list_part1 + AffWild2_image_path_list_part2
# for frame in train_img_label_fold1_list:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_train_img_path_fold1.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# train_img_label_fold1_numpy = np.concatenate((AffWild2_image_label_part1, AffWild2_image_label_part2), axis=0)
# np.savetxt(list_path_prefix + 'AffWild2_train_label_fold1.txt', train_img_label_fold1_numpy, fmt='%d')

# ###############################################################################
# with open(list_path_prefix + 'AffWild2_train_img_path_fold2.txt','w') as f:
#     u = 0
# train_img_label_fold2_list = AffWild2_image_path_list_part1 + AffWild2_image_path_list_part3
# for frame in train_img_label_fold2_list:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_train_img_path_fold2.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# train_img_label_fold2_numpy = np.concatenate((AffWild2_image_label_part1, AffWild2_image_label_part3), axis=0)
# np.savetxt(list_path_prefix + 'AffWild2_train_label_fold2.txt', train_img_label_fold2_numpy, fmt='%d')

# ###############################################################################
# with open(list_path_prefix + 'AffWild2_train_img_path_fold3.txt','w') as f:
#     u = 0
# train_img_label_fold3_list = AffWild2_image_path_list_part2 + AffWild2_image_path_list_part3
# for frame in train_img_label_fold3_list:
# 	frame_img_name = frame + '.jpg'
# 	with open(list_path_prefix + 'AffWild2_train_img_path_fold3.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
# train_img_label_fold3_numpy = np.concatenate((AffWild2_image_label_part2, AffWild2_image_label_part3), axis=0)
# np.savetxt(list_path_prefix + 'AffWild2_train_label_fold3.txt', train_img_label_fold3_numpy, fmt='%d')
