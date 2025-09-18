import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from vit import ViTForCIFAR10

class CNN_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN_classifier, self).__init__()
        # TODO
        
    def forward(self, x):
        # TODO
        return y

class MLP_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO
    
    def forward(self, x):
        # TODO
        return x

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss vs Epochs')
    plt.grid(True)
    plt.savefig('.../loss_graph.png')
    plt.close('all')


def training(model, loaders, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for _, (images,labels) in enumerate(loaders['train']):
        images = images.requires_grad_().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loaders['train'])
    return avg_loss

def valid_or_test_fn(model, loaders, criterion, device, valid_or_test):
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loaders[valid_or_test]):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            if valid_or_test == 'test':
                class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                if i in [5, 7, 10, 12, 15, 20]:
                    image = images[0,...].clone().squeeze().detach().cpu().numpy()
                    label = labels[0].item()
                    predicted_label = predicted[0].item()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(image.transpose(1,2,0))
                    plt.title(f'Predicted = {class_names[predicted_label]} / True Label = {class_names[label]}')
                    plt.savefig('.../test_out_1st_img_from_batch_{:04}.png'.format(i))
                    plt.close()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(loaders[valid_or_test])
    return avg_loss, accuracy, all_labels, all_predictions

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # TODO: Plot the confusion matrix, you can use the imported libraries above if desired

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    num_epochs = # TODO : Set a proper number of epochs (try multiple values -- should be at least 20)
    learning_rate = # TODO : Set a proper learning rate (try multiple values)
    batch_size = # TODO : Set a proper batch size (try multiple values)
    num_classes = 10
    
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    CIFAR10_data = CIFAR10(root = 'data', train = True, transform = ToTensor(), download = False)
    train_data, test_data, valid_data = torch.utils.data.random_split(CIFAR10_data, [35000, 10000, 5000])
    loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle=True)}
    
    # TODO : Define your model here
    #model = MLP_classifier(3, num_classes).to(device)
    #model = CNN_classifier(3, num_classes).to(device)
    #model = ViTForCIFAR10(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0).to(device)

    criterion = # TODO : Define your choice of loss function here, explain in report why you chose it
    optimizer = # TODO : Define your choice of optimizer here, explain in report why you chose it

    train_losses = []
    val_losses = []
    
    print('*' * 60 + '\nSTARTED Training with Batch Size = \"%s\", ' % batch_size + 'Epochs = \"%s\", ' % num_epochs + 'LR = \"%.4f\" \n' % learning_rate)
    for epoch in range(num_epochs):
        # Training
        train_loss = training(model, loaders, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation
        val_loss, valid_accuracy, _, _ = valid_or_test_fn(model, loaders, criterion, device, 'valid')
        val_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
    plot_loss(train_losses, val_losses)

    # Testing
    _, test_accuracy, all_labels, all_predictions = valid_or_test_fn(model, loaders, criterion, device, 'test')
    plot_confusion_matrix(all_labels, all_predictions, class_names)
    
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    print('\nFINISHED Training!\n' + '*' * 60)

if __name__ == "__main__":
    main()