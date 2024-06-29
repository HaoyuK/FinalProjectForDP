import argparse
import torch
import torch.optim as optim
from data_download import *
from model.cnn import *
from model.transformer import *
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy_top5(output, target):
    with torch.no_grad():
        maxk = min(5, output.size(1))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()

def test(model, test_loader, device, save_path='./save_model_path', model_type="cnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the best model
    save_model_path = os.path.join(save_path, 'best_model.pth')
    model.load_state_dict(torch.load(save_model_path))
    
    model.eval()
    all_labels = []
    all_predictions = []
    correct_test_top5 = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            correct_test_top5 += accuracy_top5(outputs, labels)
            total_test += labels.size(0)

    accuracy = accuracy_score(all_labels, all_predictions)
    top5_accuracy = correct_test_top5 / total_test
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(100)])
    cm = confusion_matrix(all_labels, all_predictions)

    print(f'Accuracy of the model on the test images: {accuracy:.4f}')
    print(f'Top-5 Accuracy of the model on the test images: {top5_accuracy:.4f}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(cm)


def main():
    parser = argparse.ArgumentParser(description='Test a trained CNN or Transformer on CIFAR-100 using config file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mean = config['data']['mean']
    std = config['data']['std']
    _, _, test_loader = get_data_loader(mean, std, batch_size=config['data']['batch_size'], num_workers=2)
    model_type = config['model']['type']

    # Initialize the model based on the config
    if config['model']['type'] == 'cnn':
        model = build_custom_cnn(
            in_channels=config['model']['in_channels'], 
            img_size=config['model']['img_size'],
            num_conv_layers=config['model']['num_conv_layers'],
            num_filters=config['model']['num_filters'],
            kernel_size=config['model']['kernel_size'],
            num_classes=config['model']['num_classes']
        )
    elif config['model']['type'] == 'transformer':
        model = VisionTransformer(
            in_channels=config['model']['in_channels'], 
            patch_size=config['model']['patch_size'],
            emb_size=config['model']['emb_size'],
            img_size=config['model']['img_size'],  
            num_heads=config['model']['num_heads'],  
            depth=config['model']['depth'],  
            num_classes=config['model']['num_classes']
        )
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")
    
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")    

    test(model, test_loader, device=config['training']['device'], save_path=config['training']['save_path'], model_type=config['model']['type'])

if __name__ == '__main__':
    main()
