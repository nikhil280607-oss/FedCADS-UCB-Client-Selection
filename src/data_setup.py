import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

# Step 1: Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading and Loading MNIST dataset")

# Step 2: Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def create_non_iid_clients(dataset, num_clients=50, shards_per_client=2, batch_size=16):
    # Step 3: Calculate shard sizes and sort data by labels
    num_shards = num_clients * shards_per_client
    imgs_per_shard = len(dataset) // num_shards
    labels = dataset.targets.numpy()
    sorted_indices = np.argsort(labels)
    shard_ids = np.arange(num_shards)
    np.random.shuffle(shard_ids)

    # Step 4: Distribute shards to create non-IID client dataloaders
    client_dataloaders = {}
    for i in range(num_clients):
        client_shards = shard_ids[i * shards_per_client : (i + 1) * shards_per_client]
        client_indices = []
        for shard in client_shards:
            start_idx = shard * imgs_per_shard
            end_idx = start_idx + imgs_per_shard
            client_indices.extend(sorted_indices[start_idx:end_idx])
        client_subset = Subset(dataset, client_indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_dataloaders[i] = client_loader
    return client_dataloaders

client_data = create_non_iid_clients(train_dataset, num_clients=50, batch_size=16)