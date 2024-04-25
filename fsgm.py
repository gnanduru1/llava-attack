import os
from getpass import getuser
import numpy as np
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageFilter

model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)

example, true_digit = mnist[0]
example.save('example.png')

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "
#target_digit = "0"
target_digit=chr(true_digit)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_id)

img = example

inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
inputs['pixel_values'].requires_grad = True

outputs = model(**inputs)
output_logits = outputs.logits

target = processor(target_digit, return_tensors="pt")
target_id = target['input_ids'][0,-1]

output_ids = torch.argmax(output_logits, dim=-1)

digit_logits = output_logits[:,-1]
digit = processor.decode(torch.argmax(digit_logits))
print("Before Attack:")
print("target digit:", target_digit, "decoded digit:", digit)

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))
print("cross entropy loss:",loss.item())

grad = torch.autograd.grad(loss, inputs['pixel_values'])[0]
epsilon = .07
nu = epsilon * torch.sign(grad)
adversarial_inputs = inputs.copy()
adversarial_inputs['pixel_values'] = inputs['pixel_values'] + nu
#adversarial_inputs['pixel_values'] = inputs['pixel_values'] - nu

outputs = model(**adversarial_inputs)
output_logits = outputs.logits

digit_logits = output_logits[:,-1]
digit = processor.decode(torch.argmax(digit_logits))
print("After Attack:")
print("target digit:", target_digit, "decoded digit:", digit)
loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))
print("cross entropy loss:",loss.item())
