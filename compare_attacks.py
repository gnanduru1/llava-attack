from attack_util import seed_everything
seed_everything()

import os, sys
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

from tqdm import tqdm
import torch
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import pandas as pd
import argparse
from attack_util import get_model_and_processor, save_img, get_mnist_instance, get_target, mnist, llava_id

model, processor = get_model_and_processor(llava_id)

def loss_fn_1(logits, target):
    return torch.nn.CrossEntropyLoss()(logits, target)


def attack1(model, inputs, target, num_iterations=100, step_size=0.01, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss_fn_1(digit_logits, target.view(-1).to(digit_logits.device))

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id == target:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure


def loss_fn_2(logits, label):
    return -torch.nn.CrossEntropyLoss()(logits, label)


def attack2(model, inputs, label, num_iterations=100, step_size=0.01, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss_fn_2(digit_logits, label.view(-1).to(digit_logits.device))

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id != label:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure

def loss1_regularized(logits, target, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, target)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    print(regularizer)
    return adv_loss + alpha*regularizer


def attack3(model, inputs, target, num_iterations=100, step_size=0.01, alpha=0.1, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss1_regularized(digit_logits, target.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id == target:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure


def loss2_regularized(logits, label, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, label)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    print(regularizer)
    return -adv_loss + alpha*regularizer


def attack4(model, inputs, label, num_iterations=100, step_size=0.01, alpha=0.1, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss2_regularized(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id != label:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure


def debug_example(i):
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    # save_img(inputs['pixel_values'], inputs['pixel_values'], f'image_{i}', #results_dir)

    target_id = get_target(model, processor, inputs, label_id)
    print(f"Attack 1&3 target = {processor.decode(target_id)}")

    result1 = attack1(model, inputs, target_id, debug=False)
    if result1 == (None, None, None):
        print("Attack 1 failed")
    _, distance_1, iters_1 = result1

    del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    print("Attack 2")
    result2 = attack2(model, inputs, label_id, debug=False)
    if result2 == (None, None, None):
        print("Attack 2 failed")
    _, distance_2, iters_2 = result2

    del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    print(f"Attack 3, target = {processor.decode(target_id)}")
    result3 = attack3(model, inputs, target_id, debug=False)
    if result3 == (None, None, None):
        print("Attack 3 failed")
    _, distance_3, iters_3 = result3

    del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    print("Attack 2")
    result4 = attack4(model, inputs, label_id, debug=False)
    if result4 == (None, None, None):
        print("Attack 4 failed")
    new_img, distance_4, iters_4 = result4

    print(f"Attack: 1 | iters: {iters_1} | distance: {distance_1}")
    print(f"Attack: 2 | iters: {iters_2} | distance: {distance_2}")
    print(f"Attack: 3 | iters: {iters_3} | distance: {distance_3}")
    print(f"Attack: 4 | iters: {iters_4} | distance: {distance_4}")
    # save_img(new_img, new_img, f'perturbed_image_{i}', results_dir)


def run_and_compare_attacks(model, mnist, id, data_range=range(3000)):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    results = pd.DataFrame(columns=[f'attack {id} distance', f'attack {id} iters'])
    
    if id==0:
        print("Attack 1")
    elif id==1:
        print("Attack 2")
    elif id==2:
        print("Attack 3")
    elif id==3:
        print("Attack 4")
    else:
        print(f"Invalid id: {id}")
        return
    
    for i in tqdm(data_range):
        row = mnist[i]
        inputs, label_id = get_mnist_instance(row, processor)
        
        if id==0:
            target_id = get_target(model, processor, inputs, label_id)
            _, distance, iters = attack1(model, inputs, target_id)
        elif id==1:
            _, distance, iters = attack2(model, inputs, label_id)
        elif id==2:
            target_id = get_target(model, processor, inputs, label_id)
            _, distance, iters = attack3(model, inputs, target_id)
        elif id==3:
            _, distance, iters = attack2(model, inputs, label_id)

        results = pd.concat([results, pd.DataFrame({'attack {id} distance': [distance], 'attack {id} iters': [iters]})])
        if (i+1)%100==0:
            print(f"Attack {id} statistics at iteration{i+1}")
            print(results.describe())
    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    
    if args.debug:
        for i in range(100):
            print(i)
            debug_example(i)
        exit()
    results_dir = 'results'

    data_dir = 'data/adv_train'
    os.makedirs(f'{results_dir}/tensors', exist_ok = True)
    os.makedirs(f'{results_dir}/images', exist_ok = True)

    results = run_and_compare_attacks(model, mnist, args.id, range(args.n))
    print(results.describe())
    results.to_csv(f'{results_dir}/compare_attacks-{args.id}.csv')
