from logging import raiseExceptions
import numpy as np
import os


#### FUNCTION TO SET YOUR PATHS

def set_your_paths(name_dataset):
    # CHANGE HERE PATHS 

    #### RAFAU
    if name_dataset == 'rafau':

        # list_path_prefix -> path of the folder where there will be stored files with the list of paths of the images and the list of AUs
        list_path_prefix = '/home/trentini/ME-GraphAU/OpenGraphAU/data/RAF-AU/list/'


        # PATH OF THE DATASET
        # this will depend on where you store the dataser

        # img_path_vita -> folder in which are stored the images
        img_path_vita = '/work/vita/datasets/RAF-AU/aligned'

        # label_root -> folder in which are stored the labels (i.e. the AUs)
        label_root = '/work/vita/datasets/RAF-AU/'

        return list_path_prefix, img_path_vita, label_root
    
    elif name_dataset == 'casme':

        # list_path_prefix -> path of the folder where there will be stored files with the list of paths of the images and the list of AUs
        list_path_prefix = '/home/trentini/ME-GraphAU/OpenGraphAU/data/CASME2/list/'


        # PATH OF THE DATASET
        # this will depend on where you store the dataser

        # img_path_vita -> folder in which are stored the images
        img_path_vita = '/work/vita/datasets/CASME2/Cropped/'

        # label_root -> file in which are stored the labels (i.e. the AUs) (here not a folder)
        label_root = '/work/vita/datasets/CASME2/CASME2-coding-20140508.xlsx'

        # split in train, val and test
        CASME2_train_subjects_split = ['/work/vita/datasets/CASME2/Cropped/sub01', '/work/vita/datasets/CASME2/Cropped/sub02', '/work/vita/datasets/CASME2/Cropped/sub04', '/work/vita/datasets/CASME2/Cropped/sub06', '/work/vita/datasets/CASME2/Cropped/sub7', '/work/vita/datasets/CASME2/Cropped/sub11', '/work/vita/datasets/CASME2/Cropped/sub12', '/work/vita/datasets/CASME2/Cropped/sub17', '/work/vita/datasets/CASME2/Cropped/sub19', '/work/vita/datasets/CASME2/Cropped/sub20', '/work/vita/datasets/CASME2/Cropped/sub21', '/work/vita/datasets/CASME2/Cropped/sub24','/work/vita/datasets/CASME2/Cropped/sub25']
        CASME2_val_subjects_split = ['/work/vita/datasets/CASME2/Cropped/sub03', '/work/vita/datasets/CASME2/Cropped/sub05', '/work/vita/datasets/CASME2/Cropped/sub16', '/work/vita/datasets/CASME2/Cropped/sub22']
        CASME2_test_subjects_split = ['/work/vita/datasets/CASME2/Cropped/sub08', '/work/vita/datasets/CASME2/Cropped/sub09', '/work/vita/datasets/CASME2/Cropped/sub10', '/work/vita/datasets/CASME2/Cropped/sub15', '/work/vita/datasets/CASME2/Cropped/sub23','/work/vita/datasets/CASME2/Cropped/sub26']


        return list_path_prefix, img_path_vita, label_root, CASME2_train_subjects_split, CASME2_val_subjects_split, CASME2_test_subjects_split

    #### AFFWILD
    elif name_dataset == 'affwild':

        # list_path_prefix -> path of the folder where there will be stored files with the list of paths of the images and the list of AUs
        list_path_prefix = '/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/' 


        # PATH OF THE DATASET
        # this will depend on where you store the dataser

        # img_path_vita -> folder in which are stored the images
        img_path_vita = '/work/vita/datasets/Aff-Wild2/cropped_aligned/'

        # label_root -> folder in which are stored the labels (i.e. the AUs)
        label_root = '/work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/'


        # names of the folders in label_root where there train_set, val_set and test_set are divided
        
        train_path ='Train_Set' 
        val_path = 'Validation_Set' 
        test_path = 'Validation_Set'   #there is no test set in our dataset

        return list_path_prefix, img_path_vita, label_root, train_path, val_path, test_path
    
    else:
        
        raise Exception('Dataset not known')