from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
import torchvision.utils as vutils
from datasets import load_dataset
from torchvision import datasets
from getpass import getuser
import random
import numpy as np

llava_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)
prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "

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
        # if debug:
        #     print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
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
    original_image.requires_grad = True
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

def loss1_regularized(logits, target, original_image, current_image, alpha=100):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, target)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    # regularizer.retain_graph = True
    # regularizer.retain_grad()
    # regularizer.backward(retain_graph=True)
    # print(regularizer)
    # adv_loss.backward(retain_graph=True)
    # print(adv_loss)
    # exit()
    return adv_loss + alpha*regularizer


def attack3(model, inputs, target, num_iterations=100, step_size=0.01, alpha=100, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    original_image
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss1_regularized(digit_logits, target.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        # if debug:
        #     print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id == target:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure


def loss2_regularized(logits, label, original_image, current_image, alpha=100):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, label)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    return -adv_loss + alpha*regularizer


def attack4(model, inputs, label, num_iterations=100, step_size=0.01, alpha=100, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    original_image.requires_grad = True
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]

        loss = loss2_regularized(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        # if debug:
        #     print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id != label:
            with torch.no_grad():
                distance = torch.nn.MSELoss()(original_image, inputs['pixel_values']).item()
            return inputs['pixel_values'], distance, i+1
    return None, None, None # failure



# def rad_loss1(logits, target, original_image, current_image, alpha=100):
#     adv_loss = torch.nn.CrossEntropyLoss()(logits, target)
#     regularizer = torch.nn.MSELoss()(original_image, current_image)
#     return adv_loss + alpha*regularizer

# def rad_loss2(logits, label, original_image, current_image, alpha=100):
#     adv_loss = torch.nn.CrossEntropyLoss()(logits, label)
#     regularizer = torch.nn.MSELoss()(original_image, current_image)
#     return -adv_loss + alpha*regularizer



# def rad_attack(model, inputs, label, num_iterations=100, step_size=0.01, alpha=100):
#     inputs['pixel_values'].requires_grad = True
#     original_image = inputs['pixel_values'].detach().clone()
#     for i in range(num_iterations):
#         output_logits = model(**inputs).logits

#         digit_logits = output_logits[:, -1]
#         loss = rad_loss(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

#         loss.backward()

#         with torch.no_grad():
#             inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
#             inputs['pixel_values'].grad.zero_()

#         digit_id = torch.argmax(digit_logits)
#         if digit_id != label:
#             break
#     return inputs['pixel_values']


def get_model_and_processor(model_id=llava_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        local_files_only=True,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def save_img(img, tensor, name, dir):
    vutils.save_image(img, f'{dir}/images/{name}.png')
    torch.save(tensor, f'{dir}/tensors/{name}.pt')

def get_mnist_instance(row, processor):
    img, label = row
    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    label_id = processor(str(label))['input_ids'][0, -1]
    return inputs, label_id

def get_mnist_dataset(processor, split='train'):
    dataset = load_dataset('mnist', split=split)
    transform = lambda img: processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    return dataset.with_transform(lambda x: {'input': transform(x['image']), 'label': x['label']})

def get_target(model, processor, inputs, label_id):
    output_logits = model(**inputs).logits
    digit_ids = processor([str(i) for i in range(10)])['input_ids'][:,-1]
    max_likelihood = 0
    target_id = 0
    for digit_id in digit_ids:
        if digit_id == label_id:
            continue
        likelihood = output_logits[0,-1,digit_id.item()]
        if likelihood >= max_likelihood:
            max_likelihood = likelihood
            target_id = digit_id
    return target_id

# def rad_attack1_debug(model, processor, inputs, target, num_iterations=100, step_size=0.01, alpha=100, debug=True):
#     inputs['pixel_values'].requires_grad = True
#     original_image = inputs['pixel_values'].detach().clone()
#     for i in range(num_iterations):
#         output_logits = model(**inputs).logits

#         digit_logits = output_logits[:, -1]
#         loss = rad_loss1(digit_logits, target.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

#         loss.backward()

#         with torch.no_grad():
#             inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
#             inputs['pixel_values'].grad.zero_()

#         digit_id = torch.argmax(digit_logits)
#         if debug:
#             print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
#         if digit_id == target:
#             return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), i+1
#     return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), num_iterations


# def rad_attack2_debug(model, processor, inputs, label, num_iterations=100, step_size=0.01, alpha=100, debug=True):
#     inputs['pixel_values'].requires_grad = True
#     original_image = inputs['pixel_values'].detach().clone()
#     for i in range(num_iterations):
#         output_logits = model(**inputs).logits

#         digit_logits = output_logits[:, -1]
#         loss = rad_loss2(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

#         loss.backward()

#         with torch.no_grad():
#             inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
#             inputs['pixel_values'].grad.zero_()

#         digit_id = torch.argmax(digit_logits)
#         if debug:
#             print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
#         if digit_id != label:
#             return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), i+1
#     return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), num_iterations

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


