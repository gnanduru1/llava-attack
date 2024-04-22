import lmql
import requests
from PIL import Image, ImageFilter

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import random

import re
from tqdm import tqdm
import json
import io
import base64

model_id = "llava-hf/llava-1.5-7b-hf"

# Load the image from the URL
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
raw_image.save('img.jpg')

# Convert the image to grayscale
img = raw_image.convert("L")

# Resize the image to 28x28 pixels
img = img.resize((28, 28), Image.Resampling.LANCZOS)

# Save the resized grayscale image to a buffer
buffered = io.BytesIO()
img.save(buffered, format="JPEG")  # Use the format appropriate for your needs; JPEG is used here for example

# Encode the bytes to Base64
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


# def image_to_ascii_bw(img):
#     # Define ASCII characters for black and white
#     ASCII_CHARS = "@ "  # '@' for black, ' ' for white


#     # Convert to grayscale
#     img = img.convert("L")

#     # Get dimensions
#     width, height = img.size

#     # Convert image to ASCII
#     ascii_art = []
#     for y in range(height):
#         line = ""
#         for x in range(width):
#             brightness = img.getpixel((x, y))
#             # Use a threshold to determine which ASCII character to use
#             if brightness < 128:
#                 line += ASCII_CHARS[0]  # Darker shades
#             else:
#                 line += ASCII_CHARS[1]  # Lighter shades
#         ascii_art.append(line)

#     return "\n".join(ascii_art)


# ascii = image_to_ascii_bw(raw_image)

# print(ascii)

@lmql.query
def chain_of_thought():
    '''lmql
sample(temperature=0.4, max_len=4096)

"{img_base64} Describe this image in 10 words or less: [DESCRIPTION]"
from
    lmql.model("llava-hf/llava-1.5-7b-hf", cuda=True)
    return ANSWER
    '''
    
print(chain_of_thought())