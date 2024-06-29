import argparse
import torch
import torch.optim as optim
from data_download import *
from model.cnn import *
from model.transformer import *  # Import your transformer model
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
import datetime

'''
usage:
python train.py --config config.yaml
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy_top5(output, target):
    with torch.no_grad():
        maxk = min(5, output.size(1))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        return correct_k

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=20, save_path='./save_model_path', model_type="cnn", cutmix_prob=0.2, alpha=1.0):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"
    model.to(device)
    
    best_accuracy = 0.0
    os.makedirs(save_path, exist_ok=True)
    save_model_path = os.path.join(save_path, 'best_model.pth')

    # Initialize TensorBoard writer
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'runs/{model_type}_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_train_top5 = 0
        
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(tqdm_loader):
            inputs, labels = inputs.to(device), labels.to(device)
        
            r = np.random.rand(1)
            if r[0] < cutmix_prob:
                inputs, (targets_a, targets_b, lam) = cutmix(inputs, labels, alpha)
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            correct_train_top5 += accuracy_top5(outputs, labels).item()
            
            tqdm_loader.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/(i+1):.4f}, Train Top-1 Acc: {correct_train / total_train:.4f}, Train Top-5 Acc: {correct_train_top5 / total_train:.4f}")

            # Log training loss and accuracy to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/train_top1', correct_train / total_train, epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/train_top5', correct_train_top5 / total_train, epoch * len(train_loader) + i)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        correct_val_top5 = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                correct_val_top5 += accuracy_top5(outputs, labels).item()
                
        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_accuracy_top5 = correct_val_top5 / total_val
        print(f'Accuracy of the model on the val images: {val_accuracy:.4f}, Top-5 Accuracy: {val_accuracy_top5:.4f}')

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val_top1', val_accuracy, epoch)
        writer.add_scalar('Accuracy/val_top5', val_accuracy_top5, epoch)

        # Save the model if it has the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_model_path)
            print(f'Saved best model with accuracy: {best_accuracy:.4f}')
        
        scheduler.step()

    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train a custom CNN or Transformer on CIFAR-100 using config file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mean = config['data']['mean']
    std = config['data']['std']
    train_loader, val_loader, test_loader = get_data_loader(mean, std, batch_size=config['data']['batch_size'], num_workers=2)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    train(model, train_loader, val_loader, test_loader, criterion, optimizer, device=config['training']['device'],
          num_epochs=config['training']['num_epochs'], save_path=config['training']['save_path'], model_type=config['model']['type'])

if __name__ == '__main__':
    main()
