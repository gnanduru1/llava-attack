# RAD Attack

This repository contains scripts attacking Meta AI's LLaVA model, and a script with a transfer attack to Salesforce's BLIP model using the regularized adversarial image perturbation method.

## Setup

We use Huggingface to download our models, and torchvision to download MNIST. The downloads will require ~16 GB of memory.

The scripts run with limited speed/effectiveness on CPU, so it is highly recommended that CUDA is installed and a GPU is available during runtime.

## Scripts

- default.py: accepts command-line arguments for model name (LLaVA or BLIP-2) and device (cpu or cuda).

- attack.py: runs the default (no regularization) gradient ascent adversarial attack on LLaVA, and saves perturbed images to results/

- rad_training.py: implements the regularized adversarial attack

- benchmark_llava.py: runs a simple validation script that passes all 10k of MNIST test images through LLaVA and verifies its output. We observe 88% accuracy.