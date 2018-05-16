# Visual Question Reasoning on General Dependency Tree
![Python version support](https://img.shields.io/badge/python-3.5%20%203.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/pytorch-0.3%200.3.1-red.svg)

This is the code for the paper on CLEVR

 **<a href="https://arxiv.org/abs/1804.00105">Visual Question Reasoning on General Dependency Tree</a>**
 <br>
Qingxing Cao,
 <a href='https://www.cs.cmu.edu/~xiaodan1/'>Xiaodan Liang</a>,
 <a href='https://bezorro.github.io/'>Bailin Li</a>,
 <a href='https://sites.google.com/site/ligb86/home/'>Guanbin Li</a>,
 <a href='http://www.linliang.net/'>Liang Lin</a>
 <br>
 Presented at [CVPR 2018](http://cvpr2018.thecvf.com/) (Spotlight Presentation)

 <div align="center">
  <img src="https://github.com/bezorro/ACMN-Pytorch/blob/master/img/introduction.png" width="450px">
</div>

If you find this code useful in your research then please cite

```
@inproceedings{cao2018acmn,
  title={Visual Question Reasoning on General Dependency Tree},
  author={Qingxing Cao and Xiaodan Liang and Bailing Li and Guanbin Li
          and Liang Lin},
  booktitle={CVPR},
  year={2018}
}
```

## Requirement
  * tensorboardX
  * skimage
  * scipy
  * numpy
  * torchvision
  * h5py
  * tqdm

## Data Preprocessing
Before you can train any models, you need to download the datasets; you also need to preprocess questions, and extract features for the images.

### Step 1: Download the data
You can download `CLEVR v1.0 (18 GB)` with the common below.
```sh
$ sh data/clevr/download_dataset.sh
```

### Step 2: Preprocess Questions
Codes for preprocessing would be available soon. For now you can download our preprocessed data with the following command:
```sh
$ sh data/clevr/download_preprocessed_questions.sh
```

### Step 3: Extract Image Features
You can extract image features with the command below.
```sh
$ sh scripts/extract_image_feature.sh
```
The extracted features `features_train.h5`, `features_val.h5`, `features_test.h5` woulde be placed in `./data/clevr/clevr_res101/`.

## Pretrained Models
You can download the pretrained models with the command below. The model will take about 2.6 GB on disk.
```sh
$ sh data/clevr/download_pretrained_model.sh
```
It is trained on `CLEVR-train` and can be validate on `CLEVR-val`.


## Training on CLEVR
You can use the `train_val.py` script to train on `CLEVR-train` and validate the model on `CLEVR-val`.
```sh
$ python scripts/train_val.py --clevr_qa_dir=data/clevr/clevr_qa_dir/ --clevr_img_h5=data/clevr/clevr_res101/
```
The below script has the hyperparameters and settings to reproduce ACMN CLEVR results.
```
$ sh scripts/train_val.sh
```

## Evaluation
You can use `train_val.py` to simply evaluate the model on `CLEVR-val` with `--no_train` option to skip the training process.
```sh
$ python scripts/train_val.py \
  --no_train=True \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --resume=data/clevr/clevr_pretrained_model.pth
```
You can use `test.py` to generate `CLEVR-test` results in `.json` format so that you can upload to CLEVR official.
```
$ python scripts/test.py \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --resume=data/clevr/clevr_pretrained_model.pth
```

## Visualizing Attention Maps
You can use `vis.py` to visualize the attention maps discribed in `Figure 4` of our paper.
```
$ python scripts/vis.py \
  --clevr_qa_dir=data/clevr/clevr_qa_dir/ \
  --clevr_img_h5=data/clevr/clevr_res101/ \
  --clevr_img_png=data/clevr/CLEVR_v1.0/ \
  --clevr_load_png=True \
  --logdir=logs/attmaps \
  --resume=data/clevr/clevr_pretrained_model.pth
```
<div align="center">
  <img src="https://github.com/bezorro/ACMN-Pytorch/blob/master/img/demo.png" width="900px">
</div>
