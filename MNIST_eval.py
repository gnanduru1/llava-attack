import requests
from PIL import Image, ImageFilter

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import random

import re
from tqdm import tqdm
import json

model_id = "llava-hf/llava-1.5-7b-hf"

prompt = """
        USER: In this task, you will be shown an image containing a single digit ranging from 0 to 9.
        Your goal is to identify the digit displayed in the image accurately. The only acceptable response
        format is a single digit, with absolutely no additional words, characters, or explanations. Just the
        digit, nothing else. I do not want any description about the color or style of the image, simply the digit present.
        Example of a correct response: "5"
        Please make sure your response adheres strictly to this format, as any other response will be incorrect.
        <image>
        ASSISTANT:
        """
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
)


processor = AutoProcessor.from_pretrained(model_id)

from tensorflow.keras.datasets import mnist
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), _ = mnist.load_data()

def get_all_digits_after_keyword(text):
    # Define the regex pattern to find the keyword "ASSISTANT:" followed by any characters and then capturing digits
    pattern = r'ASSISTANT:.*?(\d+)'
    
    # Find all non-overlapping matches in the text
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    # Convert found matches to integers and return them as a list
    return [int(digit) for match in matches for digit in match]

correct_count=0
no_digit = 0
multiple_found = 0
count = 5
results = {} 

for i in tqdm(range(count), desc="Processing MNIST images"):  # Wrap the loop with tqdm
    image = train_images[i]
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert('L')

    inputs = processor(prompt, pil_image, return_tensors='pt').to(0, torch.float16)
    print(inputs)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.decode(output[0][2:], skip_special_tokens=True)

    digits = get_all_digits_after_keyword(text)
    if len(digits)==0:
        no_digit+=1
    elif digits[0] == train_labels[i]:
        correct_count += 1
    if len(digits) > 1:
        multiple_found += 1
    # print(f"Guess was: {digit}, True label was: {train_labels[i]}")
    
    results[i] = text  # Save text to the results dictionary

# Save results to JSON file
with open('output.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"Accuracy: {correct_count / count}")
print(f"Cases where no digit was found: {no_digit}")
print(f"Cases where multiple digits were found: {multiple_found}")
