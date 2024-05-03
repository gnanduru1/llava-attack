import os
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

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
evaluate_file = 'evaluate_test2.txt'

# Process each image
for idx in range(num_images):
    image, label_index = cifar[idx]
    true_label = labels[label_index]

    prompt = f"USER: <image>\nFrom the following list of options: {', '.join(labels)}, what is this image showing?\nASSISTANT: This is a "

    inputs = processor(prompt, images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits[0, :, :]

    # Check each label's likelihood
    max_logit_score = float('-inf')
    predicted_label = None
    for label, token_ids in tokenized_labels.items():
        # Calculate average logits for each token sequence representing a label
        seq_len = outputs.logits.shape[1]
        if seq_len >= len(token_ids):
            # Get the logits for each token's position, assuming label is at the end of sequence
            relevant_logits = outputs.logits[0, -len(token_ids):, :]
            label_logits = []
            for i, token_id in enumerate(token_ids):
                token_logit = relevant_logits[i, token_id]
                label_logits.append(token_logit)
            # Calculate the mean of these logits
            average_logits = torch.tensor(label_logits).mean()
            if average_logits > max_logit_score:
                max_logit_score = average_logits
                predicted_label = label

    if predicted_label == true_label:
        correct += 1

    with open(evaluate_file, 'a') as f:
        f.write(f"Predicted: {predicted_label}, Actual: {true_label}\n")

accuracy = correct / num_images
print(f"Accuracy: {accuracy:.2f}")
with open(evaluate_file, 'a') as f:
    f.write(f"Total Correct: {correct}, Total Images: {num_images}, Accuracy: {accuracy:.2f}\n")

