from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import torchvision.utils as vutils
from torchvision import datasets
from getpass import getuser

mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)
prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "

def rad_loss(logits, label, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, label)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    return -adv_loss + alpha*regularizer


def rad_attack(model, inputs, label, num_iterations=100, step_size=0.01, alpha=0.1):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = rad_loss(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if digit_id != label:
            break
    return inputs['pixel_values']


def get_model_and_processor(model_id):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def save_img(img, tensor, name, dir):
    vutils.save_image(img, f'{dir}/images/{name}.png')
    torch.save(tensor, f'{dir}/tensors/{name}.pt')


def get_data(row, processor):
    img, label = row
    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    label_id = processor(str(label))['input_ids'][0, -1]
    return inputs, label_id

