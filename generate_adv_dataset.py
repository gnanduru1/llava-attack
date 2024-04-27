import torch
import torchvision.utils as vutils
from tqdm import tqdm
import os
from attack_util import get_data, rad_attack, get_model_and_processor, mnist

def generate_adversarial_dataset(model, processor, mnist, data_range, alpha):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    for i in tqdm(data_range):
        inputs, label_id = get_data(mnist[i], processor)
        img = rad_attack(model, inputs, label_id, alpha)
        torch.save(img, f'{data_dir}/tensors/{i}.pt')


if __name__ == "__main__":
    data_dir = 'data/rad_train'
    os.makedirs(f'{data_dir}/tensors', exist_ok = True)

    model, processor = get_model_and_processor()
    data_range = range(10000, 20000)
    alpha = 0.1
    generate_adversarial_dataset(model, processor, mnist, data_range, alpha)