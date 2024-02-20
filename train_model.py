import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Debugging tools
import smdebug.pytorch as smd


def test(model, test_loader, criterion, hook):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    running_corrects = 0
    correct = 0
    
    with torch.no_grad():  # No gradients are needed for the evaluation
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = inputs, labels = inputs.to(device), labels.to(device) # images.cuda(), labels.cuda() # 
            outputs = model(inputs)  # Forward pass
            pred = outputs.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()  # Count correct predictions
            test_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds==labels.data).item()


     # Calculate the percentage of correct answers
    test_accuracy = 100 * correct / len(test_loader.dataset)
    logger.info(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)')       
    average_accuracy = running_corrects / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Average accuracy: {100*average_accuracy}%')
    
# https://learn.udacity.com/nanodegrees/nd189/parts/cd0387/lessons/115510f6-1d53-405c-bdf9-5ee72a7a8f75/concepts/b647196d-0c39-45c3-89e7-3ab15144d643
def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook):
    count = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    for epoch in range(epochs):
        # Set the hook to train mode
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        running_loss = 0
        correct = 0
        total = 0  # Keep track of total number of samples seen

        for inputs, labels in train_loader: # Iterates through batches
            optimizer.zero_grad() # Resets gradients for new batch
            inputs, labels = inputs.to(device), labels.to(device)  # Move data/input and target/label to the device
            outputs = model(inputs)  # Runs Forward Pass
            loss = criterion(outputs, labels)  # Calculates Loss
            running_loss += loss.item()  # Sum up batch loss
            loss.backward()  # Calculates Gradients for Model Parameters
            optimizer.step() # Updates Weights
            outputs = outputs.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += outputs.eq(labels.view_as(outputs)).sum().item()  # Counts number of correct predictions
            total += labels.size(0)
            count += len(inputs)


        avg_loss = running_loss / len(train_loader)  # Average loss per batch
        avg_accuracy = 100 * correct / total  # Average accuracy
        logger.info(f'Epoch {epoch}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.2f}%')   

        hook.set_mode(smd.modes.EVAL)
        model.eval()
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
        total_accuracy = running_corrects / len(validation_loader.dataset)
        logger.info(f'Validation set: Average accuracy: {100*total_accuracy}%')
    return model    
    
    
def net():
    model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
        param.required_grad = False
    
    num_features = model.fc.in_features
    num_classes = 133 # the dataset has 133 distinct dog breeds therefore we need 133 output options
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model
    

def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                                          transforms.Resize(256),
                                                                          transforms.Resize((224, 224)),
                                                                          transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.Resize(256),
                                                                        transforms.Resize((224, 224)),
                                                                        transforms.ToTensor()])
    # Use the ImageFolder to get the data and transform it
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    # Create a loader for the training data and return it
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader, validation_data_loader
    

def main(args):
    """
    Switch to GPU if available
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))
    
    
    """
    Log the hyperparameters
    """
    logger.info(
        "batch size: {}; test batch size: {}, epochs: {}, lr: {}".format(
            args.batch_size,
            args.test_batch_size,
            args.epochs,
            args.lr
        )
    )
    
    model = net()
    model = model.to(device) # Move model to compute device
    
    loss_criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Create a hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model) # register the model
    
    train_data_loader, test_data_loader, validation_data_loader = create_data_loaders(data=args.data_dir, batch_size=args.batch_size)
    
    model = train(model, train_data_loader, validation_data_loader, args.epochs, loss_criterion, optimizer, hook)
    
    test(model, test_data_loader, loss_criterion, hook)
    
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument( "--test_batch_size", type=int, default=1000, metavar="N", help="input batch size for testing")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    
    args=parser.parse_args()
    
    main(args)