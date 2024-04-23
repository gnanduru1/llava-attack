import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageFilter

# def loss_fn(outputs, target):
#     logits = outputs.logits[:, -1, :]
#     target_ids = processor.encode(target, return_tensors='pt').to(0)
#     return F.cross_entropy(logits, target_ids.squeeze())

model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)

example = mnist[0][0]
example.save('example.png')

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT:"
target = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit 0."

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
target_inputs = processor(target, img, return_tensors='pt').to(0, torch.float16)

input_ids = inputs['input_ids']

outputs = model(**inputs)
logits = outputs.logits
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

target_string = "0"
target = tokenizer.encode(target_string, return_tensors="pt")

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output_logits.view(-1, output_logits.size(-1)), target_ids.view(-1))
print(loss)

#loss = outputs.loss
#torch.autograd.grad(loss, inputs['pixel_values'])[0]

#print(f"Loss: {loss.item()}")

