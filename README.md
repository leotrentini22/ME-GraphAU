Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition
=
This is an adaption of the paper  
> 
> **"Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition"**, 
> **IJCAI-ECAI 2022**
> 
> [[Paper]](https://arxiv.org/abs/2205.01782) [[Project]](https://www.chengluo.cc/projects/ME-AU/)
> 

<p align="center">
<img src="img/intro.png" width="70%" />
</p>

>The main novelty of the proposed approach in comparison to pre-defined AU graphs and deep learned facial display-specific graphs are illustrated in this figure.


🔧 Requirements
=
- Python 3
- PyTorch


- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```

Data and Data Prepareing Tools
=
The Dataset we used:
  * [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

We started to implement also on these Datasets:
  * [CASMEII](http://casme.psych.ac.cn/casme/e2)
  * [RAF-AU](http://whdeng.cn/RAF/model3.html)

However, training with these last two datasets needs to be done yet.

We provide tools for prepareing data in ```tool/```.
After Downloading raw data files, you can use these tools to process them, aligning with our protocals.

Firstly, you need to modify the file ```tool/dataset_utils.py```, by setting your own personal paths (more details are provided in this specific file)
Afterwards, you should run:
```
cd tool/
python dataset_process.py
python calculate_AU_class_weights.py
```

**Training with ImageNet pre-trained models**

Make sure that you download the ImageNet pre-trained models to `checkpoints/` (or you alter the checkpoint path setting in `models/resnet.py` or `models/swin_transformer.py`)

The download links of pre-trained models are in `checkpoints/checkpoints.txt`

Thanks to the offical Pytorch and [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

Training and Testing
=
- to train the first stage of our approach (ResNet-50) on AffWild2 Dataset, run:
```
python train_stage1.py --dataset AffWild2 --arc resnet50 --exp-name OpenGprahAU-ResNet50_first_stage -b 64 -lr 0.00002
```

- to test the performance on AffWild2 Dataset, run:
```
python test.py --dataset AffWild2 --resume results/OpenGprahAU-ResNet50_first_stage/bs_64_seed_0_lr_2e-05/best_model.pth --draw_text
```


### Pretrained models

AffWild2
|arch_type|GoogleDrive link| Average F1-score|
| :--- | :---: |  :---: |
|`Ours (ResNet-50)`| [link](https://drive.google.com/file/d/1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9/view?usp=sharing) | 64.7 |


📝 Main Results
=
**AffWild2**

|   Method  | AU1 | AU2 | AU4 | AU6 | AU7 | AU10 | AU12 | AU14 | AU15 | AU17 | AU23 | AU24 | Avg. |
| :-------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   EAC-Net  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 |  35.9 | 35.8 | 55.9 |
|   JAA-Net  |  47.2 | 44.0 |54.9 |77.5 |74.6 |84.0 |86.9 |61.9 |43.6 |60.3 |42.7 |41.9 |60.0|
|   LP-Net |  43.4  | 38.0  | 54.2  | 77.1  | 76.7  | 83.8  | 87.2  |63.3  |45.3  |60.5  |48.1  |54.2  |61.0|
|   ARL | 45.8 |39.8 |55.1 |75.7 |77.2 |82.3 |86.6 |58.8 |47.6 |62.1 |47.4 |55.4 |61.1|
|   SEV-Net | 58.2 |50.4 |58.3 |81.9 |73.9 |87.8 |87.5 |61.6 |52.6 |62.2 |44.6 |47.6 |63.9|
|   FAUDT | 51.7 |49.3 |61.0 |77.8 |79.5 |82.9 |86.3 |67.6 |51.9 |63.0 |43.7 |56.3 |64.2 |
|   SRERL | 46.9 |45.3 |55.6 |77.1 |78.4 |83.5 |87.6 |63.9 |52.2 |63.9  |47.1 |53.3 |62.9 |
|   UGN-B | 54.2  |46.4  |56.8  |76.2  |76.7  |82.4  |86.1  |64.7  |51.2  |63.1  |48.5  |53.6  |63.3 |
|   HMP-PS | 53.1 |46.1 |56.0 |76.5 |76.9 |82.1 |86.4 |64.8 |51.5 |63.0 |49.9 | 54.5  |63.4 |
|   Ours (ResNet-50) | 53.7 |46.9 |59.0 |78.5 |80.0 |84.4 |87.8 |67.3 |52.5 |63.2 |50.6 |52.4 |64.7 |
|   Ours (Swin-B) | 52.7 |44.3 |60.9 |79.9 |80.1| 85.3 |89.2| 69.4| 55.4| 64.4| 49.8 |55.1 |65.5|


🎓 Citation
=
if the code or method help you in the research, please cite the following paper:
```
@article{luo2022learning,
title = {Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition},
author = {Luo, Cheng and Song, Siyang and Xie, Weicheng and Shen, Linlin and Gunes, Hatice},
journal={arXiv preprint arXiv:2205.01782},
year={2022}
}


@inproceedings{luo2022learning,
  title     = {Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition},
  author    = {Luo, Cheng and Song, Siyang and Xie, Weicheng and Shen, Linlin and Gunes, Hatice},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  pages     = {1239--1246},
  year      = {2022}
}

```
