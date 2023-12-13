import numpy as np 
import torch.nn as nn

from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def evaluate_multitasking_ability(architectures, tasks, evaluate_function):
    """
    Evaluate the multi-tasking ability of each architecture.

    Parameters:
    architectures (list): List of architectures to evaluate.
    tasks (list): List of tasks or datasets to test each architecture on.
    evaluate_function (function): A function that evaluates an architecture on a given task.

    Returns:
    dict: A dictionary with architecture as keys and their multitasking scores as values.
    """
    multitasking_scores = {}

    for architecture in architectures:
        task_scores = []
        for task in tasks:
            score = evaluate_function(architecture, task)
            task_scores.append(score)
        multitasking_scores[architecture] = np.mean(task_scores)

    return multitasking_scores

def evaluate_function(architecture, task):
    """
    Evaluate the given architecture on a specific task.

    Parameters:
    architecture: A representation of the neural network architecture.
    task: Information about the task or dataset to evaluate the architecture on.

    Returns:
    float: Performance score of the architecture on the given task.
    """

    # Setup the architecture based on the provided parameters
    model = setup_model(architecture)

    # Load the dataset for the task
    data = load_dataset(task)

    # Train the model on the dataset
    train_model(model, data)

    # Evaluate the model on the validation or test set
    performance_score = test_model(model, data)

    return performance_score

def setup_model(architecture):
    # Example: architecture = {'num_layers': 3, 'layer_type': 'linear', 'activation': 'relu'}

    layers = []
    input_size = 784  # Example input size, e.g., for MNIST dataset
    for _ in range(architecture['num_layers']):
        layers.append(nn.Linear(input_size, 128))
        if architecture['activation'] == 'relu':
            layers.append(nn.ReLU())
        elif architecture['activation'] == 'sigmoid':
            layers.append(nn.Sigmoid())
        # Add more activation functions as needed
        input_size = 128  # Set next layer's input size

    layers.append(nn.Linear(128, 10))  # Example output layer for a classification task
    model = nn.Sequential(*layers)
    return model



def load_dataset(task):
    # Example: task = 'MNIST'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    if task == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Add more datasets as needed

    return trainset, testset


def train_model(model, data, epochs=5, learning_rate=0.001):
    trainset, _ = data
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} - Training loss: {running_loss/len(trainloader)}")


def test_model(model, data):
    _, testset = data
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    performance_score = 100 * correct / total
    return performance_score

