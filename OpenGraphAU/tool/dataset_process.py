
import os
import pandas as pd
import numpy as np
from dataset_utils import *
import os
import random

AUs = ['1', '2' ,'4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22' ,'23', '24', '25', '26', '27', '32', '38', '39']
mcro_AUs = ['L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
total_AUs = AUs+mcro_AUs

new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []


# #-------------------------------------------------------------------------------------------------------------------------------
# #RAF-AU
print("processing RAF-AU------------------------------------------------------------")

name_dataset='rafau'

au_idx = ['1', '2' ,'4', '5', '6', '7', '9', '10', '12', '14', '15', '16', '17', '18', '19' ,'20', '22', '23', '24', '25', '26', '27', '32', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6','L10','R10', 'L12', 'R12', 'L14', 'R14']

list_path_prefix, img_path_vita, label_root = set_your_paths(name_dataset)


with open(os.path.join(label_root,'RAFAU_label.txt'), 'r') as f:
    label_lines = f.readlines()

img_path_list = []
au_label_list = []
for idx in range(0,2000):
    train_au_item = label_lines[idx]
    items = train_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

TRAIN_numpy_list = np.concatenate(au_label_list,axis=0)
RAF_AU_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) -1

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_train_label.txt'), RAF_AU_train_image_label,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(RAF_AU_train_image_label)


with open(os.path.join(list_path_prefix, 'RAF_AU_train_img_path.txt'), 'a+') as f:
    l=0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join(img_path_vita,img_path+'\n'))
        new_dataset_train_img_list.append(os.path.join(img_path_vita,img_path+'\n'))


img_path_list = []
au_label_list = []
for idx in range(2001,2500):
    val_au_item = label_lines[idx]
    items = val_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

VAL_numpy_list = np.concatenate(au_label_list,axis=0)

RAF_AU_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) -1

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_val_label.txt'), RAF_AU_val_image_label,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(RAF_AU_val_image_label)


with open(os.path.join(list_path_prefix, 'RAF_AU_val_img_path.txt'), 'a+') as f:
    l=0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join(img_path_vita,img_path+'\n'))
        new_dataset_val_img_list.append(os.path.join(img_path_vita,img_path+'\n'))


img_path_list = []
au_label_list = []
for idx in range(2501,3700):
    test_au_item = label_lines[idx]
    items = test_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

TEST_numpy_list = np.concatenate(au_label_list,axis=0)
RAF_AU_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) -1

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_test_image_label[:, index] = TEST_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_test_label.txt'),  RAF_AU_test_image_label,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(RAF_AU_test_image_label)

with open(os.path.join(list_path_prefix, 'RAF_AU_test_img_path.txt'), 'a+') as f:
    l=0

for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join(img_path_vita,img_path+'\n'))
        new_dataset_test_img_list.append(os.path.join(img_path_vita,img_path+'\n'))

# #-------------------------------------------------------------------------------------------------------------------------------
# #CASME2
# print("processing CASME2------------------------------------------------------------")

name_dataset = 'casme'


# to delete if there is will to train on all the datasets together
new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []
#until here

au_ids  = ['1', '2' ,'4', '5', '6', '7', '9', '10','12', '14', '15','17', '18', '20', '24', '25', '26', '38' ,'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']



list_path_prefix, img_path_vita, label_root, CASME2_train_subjects_split, CASME2_val_subjects_split, CASME2_test_subjects_split = set_your_paths(name_dataset)


df = pd.read_excel(label_root)
all_list = []


df.iloc[:, 0] =  df.iloc[:, 0].astype(str)
df = df.iloc[:,[0,1,3,5,7]]
values = df.values

train_img_path_list = []
train_au_label_list = []

val_img_path_list = []
val_au_label_list = []

test_img_path_list = []
test_au_label_list = []

for line in values:
    subject = os.path.join(img_path_vita,'sub'+ line[0].zfill(2))
    sequence = line[1]
    OnsetFrame = line[2]
    OffsetFrame = line[3]
    au = str(line[4])
    flag = 0
    au_label = np.zeros((1,len(au_ids)))
    if au !='?':
        au_items = au.split('+')
        for item in au_items:
            # print(item)
            if item in au_ids:
                flag=1
                au_label[0, au_ids.index(item)] = 1

    if flag>0:
        for i in range(OnsetFrame, OffsetFrame+1):
            img_path = os.path.join(subject,str(sequence),'reg_img'+ str(i) +'.jpg')
            if subject in CASME2_train_subjects_split:
                train_img_path_list.append(img_path)
                train_au_label_list.append(au_label)
            elif subject in CASME2_val_subjects_split:
                val_img_path_list.append(img_path)
                val_au_label_list.append(au_label)
            else:
                test_img_path_list.append(img_path)
                test_au_label_list.append(au_label)

TRAIN_numpy_list = np.concatenate(train_au_label_list,axis=0)
VAL_numpy_list = np.concatenate(val_au_label_list,axis=0)
TEST_numpy_list = np.concatenate(test_au_label_list,axis=0)


CASME2_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CASME2_train_label.txt'),  CASME2_train_image_label, fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(CASME2_train_image_label)


with open(os.path.join(list_path_prefix, 'CASME2_train_img_path.txt'), 'w+') as f:
    i=0
for img_path in train_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_train_img_list.append(os.path.join('CASME2', img_path+'\n'))



CASME2_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CASME2_val_label.txt'),  CASME2_val_image_label, fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(CASME2_val_image_label)

with open(os.path.join(list_path_prefix, 'CASME2_val_img_path.txt'), 'w+') as f:
    i=0
for img_path in val_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_val_img_list.append(os.path.join('CASME2', img_path+'\n'))


CASME2_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_test_image_label[:, index] = TEST_numpy_list[:, i]




np.savetxt(os.path.join(list_path_prefix,'CASME2_test_label.txt'),  CASME2_test_image_label, fmt='%d', delimiter=' ')

with open(os.path.join(list_path_prefix, 'CASME2_test_img_path.txt'), 'w+') as f:
    i=0
for img_path in test_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_test_img_list.append(os.path.join('CASME2', img_path+'\n'))
new_dataset_test_label_list.append(CASME2_test_image_label)


#-------------------------------------------------------------------------------------------------------------------------------
# AffWild2
print("processing AffWild2------------------------------------------------------------")

name_dataset = 'affwild'

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']



# to delete if there is will to train on all the datasets together
new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []
#until here

list_path_prefix, img_path_vita, label_root, train_path, val_path, test_path = set_your_paths(name_dataset)



train_list = os.listdir(os.path.join(label_root, train_path)) 

train_labels = os.path.join(list_path_prefix, 'AffWild2_train_label.txt')
with open(train_labels, 'w') as  f:
    i = 0

val_list = os.listdir(os.path.join(label_root, val_path))

val_labels = os.path.join(list_path_prefix, 'AffWild2_val_label.txt')
with open(val_labels, 'w') as  f:
    i = 0

test_list = os.listdir(os.path.join(label_root, test_path)) 

test_labels = os.path.join(list_path_prefix, 'AffWild2_test_label.txt') 
with open(test_labels, 'w') as  f:    
    i = 0


train_img_path = os.path.join(list_path_prefix, 'AffWild2_train_img_path.txt')
with open(train_img_path, 'w') as f:
    i = 0
val_img_path = os.path.join(list_path_prefix, 'AffWild2_val_img_path.txt')
with open(val_img_path, 'w') as f:
    i = 0
test_img_path = os.path.join(list_path_prefix, 'AffWild2_test_img_path.txt') 
with open(test_img_path, 'w') as f:    
    i = 0




# Train

au_labels = []
au_img_path = []
for train_txt in train_list:  #scorre le cartelle in train list
    with open(os.path.join(os.path.join(label_root, train_path), train_txt), 'r') as f: #apre la cartella
        lines = f.readlines()  #legge contenuto
    lines = lines[1:] #toglie prima linea
    for j, line in enumerate(lines):   #converte contenuto linee
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:  #se c'e' -1, skippa linea
            continue
        au_labels.append(line.reshape(1, -1)) #appende la linea rishapata
        au_img_path.append(os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(train_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg'))  #appende il path dell'immagine, "split" splitta il path dove ci sono i punti e poi prende solo cio che viene prima del punto (quindi toglie ".txt")


au_labels = np.concatenate(au_labels, axis=0)
AffWild2_train_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_train_image_label[:, index] = au_labels[:, i]

with open(train_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))
        new_dataset_train_img_list.append(os.path.join('AffWild2',line+'\n'))

np.savetxt(train_labels, AffWild2_train_image_label ,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(AffWild2_train_image_label)




# Validation

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
        au_img_path.append(os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(val_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AffWild2_val_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_val_image_label[:, index] = au_labels[:, i]

with open(val_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))
        new_dataset_val_img_list.append(os.path.join('AffWild2',line+'\n'))

np.savetxt(val_labels, AffWild2_val_image_label ,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(AffWild2_val_image_label)





# Test

au_labels = []
au_img_path = []             # for loop not clear  ->   test_text is a number? Or it's just a loop over the folders
for test_txt in test_list:   # test list = directories in /work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/Validation_Set
    with open(os.path.join(os.path.join(label_root, test_path), test_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(test_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg'))    # what does it do here?


au_labels = np.concatenate(au_labels, axis=0)
AffWild2_test_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_test_image_label[:, index] = au_labels[:, i]

with open(test_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))    #
        new_dataset_test_img_list.append(os.path.join('AffWild2',line+'\n'))     # what does it do here?

np.savetxt(test_labels, AffWild2_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(AffWild2_test_image_label)


#
# print(len(new_dataset_train_img_list))
# print(len(new_dataset_val_img_list))
# print(len(new_dataset_test_img_list))

new_dataset_train_label_list = np.concatenate(new_dataset_train_label_list, axis=0)
new_dataset_val_label_list = np.concatenate(new_dataset_val_label_list, axis=0)
new_dataset_test_label_list = np.concatenate(new_dataset_test_label_list, axis=0)


sub_list = [0,1,2,4,7,8,11]

for i in range(new_dataset_train_label_list.shape[0]):
    for j in range(27, 12):  #qua "12" dipende dal numero di AUs che abbiamo  (anche sotto)
        sub_au_label = new_dataset_train_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_train_label_list[i, main_au_index] = 1


for i in range(new_dataset_val_label_list.shape[0]):
    for j in range(27, 12):
        sub_au_label = new_dataset_val_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_val_label_list[i, main_au_index] = 1

for i in range(new_dataset_test_label_list.shape[0]):
    for j in range(27, 12):
        sub_au_label = new_dataset_test_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_test_label_list[i, main_au_index] = 1



np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_train_label_list ,fmt='%d', delimiter=' ')
np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_val_label_list ,fmt='%d', delimiter=' ')
np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_test_label_list ,fmt='%d', delimiter=' ')

with open(os.path.join(list_path_prefix, 'AffWild2_train_img_path.txt') , 'w+') as f:
    for line in new_dataset_train_img_list:
        f.write(line)

with open(os.path.join(list_path_prefix, 'AffWild2_val_img_path.txt') , 'w+') as f:
    for line in new_dataset_val_img_list:
        f.write(line)

with open(os.path.join(list_path_prefix, 'AffWild2_test_img_path.txt') , 'w+') as f:
    for line in new_dataset_test_img_list:
        f.write(line)