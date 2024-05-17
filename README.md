# RAD Attack

We develop 4 white-box adversarial attack algorithms causing Meta AI's LLaVA to misclassify MNIST and CIFAR-10 images. We develop two primary loss functions that we optimize with gradient descent until the model misclassifies the image. We develop regularized versions of these loss functions that add a weighted term for the deviation of the perturbed image. This repository contains scripts attacking Meta AI's LLaVA model, tuning attack hyperparameters, comparing different attacks, demonstrating transfer attack on Salesforce's BLIP model, and finetuning LLaVA with adversarially perturbed images.

## Setup

We use Hugging Face to download our models, and Hugging Face and torchvision to download MNIST and CIFAR. The downloads will require ~16 GB of memory.

The scripts run with limited speed/effectiveness on CPU, so it is highly recommended that CUDA is installed and a GPU is available during runtime.

To activate our environment:
```
export HF_TOKEN="your_hugging_face_api_token"
source env.sh
```

## Scripts

- [attack.py](attack.py): Runs the default (no regularization) gradient descent adversarial attack on LLaVA with MNIST image and saves perturbed image to results/

- [attack_cifar.py](attack_cifar.py): Runs default attack on LLaVA with image from CIFAR-10 dataset

- [benchmark_llava.py](benchmark_llava.py): runs a simple validation script that passes all 10k of MNIST test images through LLaVA and verifies its output. We observe 88% accuracy.

- [tune_alpha.py](tune_alpha.py): Runs a grid search to tune the hyperparameter associated with the regularizer term in attacks 3 & 4 and outputs summary statistics in results/tune_alpha-{id}.csv

- [compare_attacks.py](compare_attacks.py): Runs an attack on a subset of MNIST and outputs summary statistics in results/compare_attacks-{id}.csv

- [analysis.py](analysis.py): Analyzes and graphs the results of the attack comparison and  hyperparameter tuning

- [generate_rad_dataset.py](generate_rad_dataset.py): Generates a dataset of adversarially perturbed MNIST images and stores the images as tensors in rad_data/tensors-{id}/

- [training.py](training.py): Finetunes LLaVA on MNIST images and/or perturbed MNIST images and outputs results to results/{rad or reg}-train-{id}.csv

## Slurm Scripts

We develop a number of slurm scripts to run our experiments on the University of Virginia's Rivanna HPC.

- [tune_alpha.slurm](tune_alpha.slurm): Runs hyperparameter tuning experiment for each attack

- [compare_attacks.slurm](compare_attacks.slurm): Runs attack comparison experiment for each attack

- [generate_rad_dataset.slurm](generate_rad_dataset.slurm): Runs adversarial dataset generation for each attack

- [training.slurm](training.slurm): Runs a baseline experiment finetuning LLaVA on regular MNIST images

- [rad_training.slurm](rad_training.slurm): Run adversarial finetuning experiment by finetuning LLaVA on both MNIST images and perturbed MNIST images for each attack

## Write-Up

Our report is uploaded to the repository in [report.pdf](report.pdf)
