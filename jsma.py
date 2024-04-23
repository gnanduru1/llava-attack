import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageFilter

model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)

example = mnist[0][0]
example.save('example.png')

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit 5."
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

target_inputs = processor(target, img, return_tensors='pt').to(0, torch.float16)

input_ids = inputs['input_ids']
input_embeddings = model.get_input_embeddings()(input_ids.to(torch.int64))
#input_embeddings.requires_grad = True  # Enable gradient computation for input embeddings

outputs = model(inputs_embeds=input_embeddings, labels=target_inputs['input_ids'])

loss = outputs.loss

# Compute gradients with respect to input embeddings
input_gradients = torch.autograd.grad(loss, input_embeddings)[0]

print(f"Loss: {loss.item()}")

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# Print the extracted input gradients
print(f"Input gradients: {input_gradients}")
