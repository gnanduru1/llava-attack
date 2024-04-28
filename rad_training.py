#seed everything
#train one model on 2 epochs of n mnist examples
#train the other on 1 epoch of the same n mnist examples, then one epoch on those examples perturbed
#evaluate both models on the validation dataset

import os
import glob
import torchvision.utils as vutils
from torchvision import datasets
from attack_util import seed_everything, get_model_and_processor, get_mnist_instance, get_mnist_dataset, mnist
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
seed_everything()

def train(model, data, num_epochs=1, batch_size=32):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def get_rad_data(rad_data_dir, processor):
    file_list = glob.glob(os.path.join(rad_data_dir, '*.pt'))
    rad_data = []
    labels = []
    for file in file_list:
        print(file)
        index, label = file.split('/')[-1].split('-')
        rad_data.append(torch.load(file))
        labels.append(processor(label)['input_ids'][0,-1])
    return rad_data

def evaluate(model):
    pass

# def get_mnist_dataset(processor, mnist, data_range):
#     data = [get_mnist_instance(mnist[i], processor) for i in data_range]
#     x = [row[0] for row in data]
#     y = [row[1] for row in data]
#     dataset_dict = {"data": x, "label": y}
#     hf_dataset = Dataset.from_dict(dataset_dict).with_format("torch")
#     return DataLoader(hf_dataset, batch_size=32, shuffle=True)



if __name__ == '__main__':
    # create command line argument for rad or not
    # if rad, train on first n examples, then train on rad examples, then evaluate
    # if not rad, train on first 2*n examples, then evaluate
    model, processor = get_model_and_processor()
    n = "20000"
    batch_size = 32
    rad = False
    rad_data_dir = "data/"
    mnist_data = get_mnist_dataset(processor, split=f'train[:{n}]')
    # if rad:
    #     train(model, mnist_data, num_epochs=1, batch_size=batch_size)
    #     rad_data = get_rad_data(rad_data_dir)
    #     train(model, rad_data, num_epochs=1, batch_size=batch_size)
    #     # rad_train(model, num_epochs=1)
    # else:
    #     train(model, mnist_data, num_epochs=2, batch_size=batch_size)
