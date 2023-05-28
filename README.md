<div id="top"></div>

<br />
<div align="center">
<h1 align="center">Learning Facial Action Unit Recognition</h1>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#General-Information">General Information</a></li>
    <li><a href="#Our-Team">Our Team</a></li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Structure">Structure</a></li>
    <li><a href="#Documentation">Documentation</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#Results">Results</a></li>
  </ol>
</details>

## General Information

The repository contains the code and report for the first part of the Facial Action Unit Recognition project. In particular, this is an adaption to our datasets of the paper  
> 
> **"Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition"**, 
> **IJCAI-ECAI 2022**
> 
> [[Paper]](https://arxiv.org/abs/2205.01782) [[Project]](https://www.chengluo.cc/projects/ME-AU/)
> 

## Requirements
=
- Python 3
- PyTorch


- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```

## Data and preprocessing

The Dataset we used:
  * [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

We started to implement also on these Datasets:
  * [CASMEII](http://casme.psych.ac.cn/casme/e2)
  * [RAF-AU](http://whdeng.cn/RAF/model3.html)

However, training with these last two datasets is still not done.

We provide tools for prepareing data in ```tool/```.

1. Download raw data files
2. Modify the file ```tool/dataset_utils.py```, by setting your own personal paths (e.g. the path where you store the dataset, or the path where you would like to store the lists of AUs. More details are provided in this specific file)
3. From this folder, run:
   ```
   cd tool/
   python dataset_process.py
   python calculate_AU_class_weights.py
   ```
## Training with ImageNet pre-trained models

Make sure that you download the ImageNet pre-trained models to `checkpoints/` (or you alter the checkpoint path setting in `models/resnet.py` or `models/swin_transformer.py`)

The download links of pre-trained models are in `checkpoints/checkpoints.txt`

Thanks to the offical Pytorch and [Swin Transformer](https://github.com/microsoft/Swin-Transformer)


## Training and Testing

- to train the first stage of our approach (using ResNet-50 as backbone) on AffWild2 Dataset, run:
```
python train_stage1.py --dataset AffWild2 --arc resnet50 --exp-name OpenGprahAU-ResNet50_first_stage -b 64 -lr 0.00002
```

- to test the performance on AffWild2 Dataset, run:
```
python test.py --dataset AffWild2 --resume results/OpenGprahAU-ResNet50_first_stage/bs_64_seed_0_lr_2e-05/best_model.pth --draw_text
```

We provide a release of our trained model. If you want to use it, please download it and run the commands above by setting the correct path of the model after `--resume`

### Our trained models

AffWild2
|arch_type|GoogleDrive link| Average F1-score|
| :--- | :---: |  :---: |
|`Ours (ResNet-50)`| [link](https://drive.google.com/file/d/1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9/view?usp=sharing) | 48.28 |


## Main Results

As a final result, we obtained an average f1-score of **48.23** on the test set

**AffWild2**

|   Method  | AU1 | AU2 | AU4 | AU6 | AU7 | AU10 | AU12 | AU15 | AU15 | AU23 | AU24 | AU25 | AU26 | Avg. |
| :-------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   Ours (ResNet-50) | 51.27 | 26.72 | 48.10 | 57.52 | 73.25 | 73.19 | 69.06 | 25.27 | 13.54 | 7.81 | 83.96 | 22.31 | 48.28 |

<p align="right">(<a href="#top">Back to top</a>)</p>



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
