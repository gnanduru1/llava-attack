import os
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm

# Set environment for datasets
user = os.getenv('USER')
dataset_path = f'/scratch/{user}/datasets'

# Load CIFAR-10 dataset
cifar = datasets.CIFAR10(dataset_path, train=False, download=True)

# Initialize model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(model_id)

# Setup for evaluation
num_images = len(cifar)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
tokenized_labels = {label: processor.tokenizer(label, add_special_tokens=False)['input_ids'] for label in labels}
correct = 0
count=0
evaluate_file = 'evaluate_test3.txt'

def moving_average(x, tensor):
    result = []
    for i in range(x):
        # Compute mean from index i to tensor.shape[0]-x+i
        mean_val = tensor[i:tensor.shape[0]-x+i].mean(dim=0)
        result.append(mean_val)
    return torch.stack(result)

# Process each image
for idx in tqdm(range(num_images), desc='Evaluating Images'):
    image, label_index = cifar[idx]
    true_label = labels[label_index]

    prompt = f"USER: <image>\nFrom the following list of options: {', '.join(labels)}, what is this image showing?\nASSISTANT: This is a "

    inputs = processor(prompt, images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits



    # Determine predicted label by finding the best match with known labels
    max_logit_score = float('-inf')
    predicted_label = None
    for label, token_ids in tokenized_labels.items():
        # print(f"Label: {label}, ids: {token_ids}")
        sequence_length = logits.shape[1]
        if sequence_length >= len(token_ids):
            
            relevant_logits = moving_average(len(token_ids), logits[0])
            label_logits = torch.tensor([relevant_logits[idx, token_id] for idx, token_id in enumerate(token_ids)])
            # print(label_logits)
            average_logit_score = label_logits.mean()
            if average_logit_score > max_logit_score:
                max_logit_score = average_logit_score
                predicted_label = label

    if predicted_label == true_label:
        correct += 1
    count += 1
    
    with open(evaluate_file, 'a') as f:
        f.write(f"Predicted: {predicted_label}, Actual: {true_label}\n")
    print(f"Total Correct: {correct}, Total Images: {count}, Accuracy: {correct / count:.2f}\n")

accuracy = correct / num_images
print(f"Accuracy: {accuracy:.2f}")
with open(evaluate_file, 'a') as f:
    f.write(f"Total Correct: {correct}, Total Images: {num_images}, Accuracy: {accuracy:.2f}\n")

