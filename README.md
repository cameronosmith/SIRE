# 👑 SIRE: SE(3) Intrinsic Rigidity Embeddings
(TMLR)

[Project Page](https://cameronosmith.github.io/sire),
[arXiv](https://arxiv.org/abs/2503.07739)

<img src="https://cameronosmith.github.io/sire/img/overview.png" style="width:50%;">

## Training Demo

We provided a sample of the dataset here in `sample_data` and a checkpoint as well. Run the training script (should just open a wandb script) with the command `python train.py -b 4 -n dogs_training_demo --until_img 200 -c dogs_ckpt.pt`. 

Still TODO is to add the dataset preprocessing script to process video directories in to this format.
