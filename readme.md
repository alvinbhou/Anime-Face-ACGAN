# Anime-Face-ACGAN
## Info
An ACGAN to generate anime faces with specific hair and eyes color.

## Dataset
Special thanks for 樊恩宇, TA of MLDS providing data. Data and tags are originally from konachan.


## Usage
ACGAN_train.py

```
usage: ACGAN_train.py [-h] --uid UID [--train_path TRAIN_PATH]
                      [--gen_lr GEN_LR] [--dis_lr DIS_LR]
                      [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                      [--latent LATENT]

Anime ACGAN

optional arguments:
  -h, --help            show this help message and exit
  --uid UID             training uid
  --train_path TRAIN_PATH
                        training data path
  --gen_lr GEN_LR       learning rate of generator
  --dis_lr DIS_LR       learning rate of discriminator
  --batch_size BATCH_SIZE
                        batch size for training
  --epochs EPOCHS       epochs for training
  --latent LATENT       latent size
```

generate.py

```
python generate.py pretrained_models/sample.txt
```

This will generate images with the condition given in [sample.txt](pretrained_models/sample.txt)

## Images

![](img/demo.gif)

<img src = "img/demo.png" width="200px">





