# NoisyViT_Food
Improving Food Image Recognition  with Noisy Vision Transformer

In this study, we investigate the potential of Noisy Vision Transformers (NoisyViT) for improving food classification performance. By introducing noise into the learning process, NoisyViT reduces task complexity and adjusts the entropy of the system, leading to enhanced model accuracy

<p align="left"> 
<img width="800" src="https://github.com/Tonmoy-Ghosh/NoisyViT_Food/blob/main/Framework.png">
</p>

### Environment (Python 3.8.12)
```
# Install required packages (see requirements.txt)
pip install -r requirements.txt

# Docker file is for runing code in docker
# batch file for runing docker in WSL
```
### Datasets:

- Download the [Food2k](http://123.57.42.89/FoodProject.html) dataset.

- After rearranging with 'run_create_dataset_for_NoisyViT_food.py', The Food2k folder has a structure like this:
```
food2K/
├── train/
│   ├── 0/
│   │   ├── 148.JPEG
│   │   ├── 149.JPEG
│   │   └── ...
│   ├── 1/
│   └── ...
│   └── 1999/
├── val/
│   ├── 0/
│   │   ├── 177.JPEG
│   │   ├── 204.JPEG
│   │   └── ...
│   ├── 1/
│   └── ...
│   └── 1999/
├── test/
│   ├── 0/
│   │   ├── 160.JPEG
│   │   ├── 167.JPEG
│   │   └── ...
│   ├── 1/
│   └── ...
│   └── 1999/
```
- Download the [Food101](https://www.kaggle.com/datasets/srujanesanakarra/food101) dataset.

- Download the [CNFOOD241](https://www.kaggle.com/datasets/zachaluza/cnfood-241) dataset.

### Pretrained ViT
NoisyViT-B_16-384/NoisyViT-B_16-224 (pre-trained on ImageNet-1K) performance on food recognition dataset:
| Model  | Dataset | Top1 Acc | Top5 Acc | Download|
| ------ | ------- | -------- | -------- | ------- |
| NoisyViT-B_16-384 | Food2k | 95.0% | 99.8% | [Food2k384_model](https://alabama.box.com/s/fp98muxob90ais8tr3x88c9yh7cd1vm2) |
| NoisyViT-B_16-224 | Food2k | 94.1% | 99.8% | [Food2k224_model](https://alabama.box.com/s/2t43rxidr4tgpqk4ffcqh2q1b3hg0vcx) |
| NoisyViT-B_16-224 | Food101 | 99.5% | 100% | [Food101_model](https://alabama.box.com/s/iviqqnxssr26x16uedvkpgd0b6ij33mg) |
| NoisyViT-B_16-224 | CNFOOD241 | 96.6% | 99.9% | [CNFOOD241_model](https://alabama.box.com/s/r4abnbfch0qo0ecwa97o1v7yivlsqvae) |


### Training:

Commands can be found in `runScript.txt`. An example:
```
python Main.py --lr 0.00001 --epochs 30 --batch_size 16 --layer 11 --gpu_id 0 --res 224 --patch_size 16 --scale base --noise_type linear --datasets food2k --num_classes 2000 --tra 0 --inf 1 --OptimalQ 1
```

### Inference:

Commands can be found in `runScript.txt`. An example:
```
python Inference.py --lr 0.00001 --epochs 1 --batch_size 16 --layer 11 --gpu_id 0 --res 224 --patch_size 16 --scale base --noise_type linear --model_saved_path 'acc_0.9325_lr_1e-05_bs_16_layer_11_base_224_16_linear_food2k_NoisyViT.pkl' --test_path './test' --num_classes 2000 --tra 0 --inf 1 --OptimalQ 1
```

## Acknowledgement 
Our code is largely borrowed from [NoisyNN](https://github.com/Shawey94/NoisyNN/tree/main)

The code was developed by [Tonmoy Ghosh](https://scholar.google.com/citations?user=O-8G6JEAAAAJ&hl=en) at [CLAWS](https://claws.eng.ua.edu/research-projects/9-research-projects/302-improving-food-image-recognition-with-noisy-vision-transformer) lab, the University of Alabama.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
