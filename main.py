import argparse
import os
from random import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from data import data_transforms
from model import Classifier


def train(epoch):
    global itrs_train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        itrs_train += 1 
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.data.item(), itrs_train)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def validation():
    global itrs_val
    global num_val
    model.eval()


    
    validation_loss = 0
    val_correct = 0
    num_val+=1
    for data, target in val_loader:
        itrs_val+=1
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        writer.add_scalar('Loss/validation', loss.data.item(), itrs_val)
        validation_loss += loss.data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        val_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    validation_loss /= len(val_loader.dataset)
    writer.add_scalar('Accuracy/validation', val_correct, num_val)
    writer.add_scalar('Average Loss/validation', validation_loss, num_val)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, val_correct, len(val_loader.dataset),
        100. * val_correct / len(val_loader.dataset)))

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        train_loss += loss.data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    train_loss /= len(train_loader.dataset)
    writer.add_scalar('Accuracy/train', train_correct, num_val)
    writer.add_scalar('Average Loss/train', train_loss, num_val)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))


def tupletype(s):
    try:
        x, y,= map(float, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("betas must be a two-elements tuple")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument("--finetune", action='store_true', default=False,
                        help="train the whole model or just the classifier")
                        
    #Choice of encoder model
    parser.add_argument('--encoder', type=str, default='vgg16', choices=['vgg16', 'resnet18', 'resnet50', 'inception_v3'],
                        help='encoder (default: vgg16)')

    #Choice of optimizer and hyperparameters for the chosen optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Adagrad'],
                        help='optimizer (default: SGD)')
    parser.add_argument('--lr_SGD', type=float, default=0.1, metavar='LRSGD',
                        help='learning rate for SGD (default: 0.1)')
    parser.add_argument('--momentum_SGD', type=float, default=0.5, metavar='MSGD',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weightdecay_SGD', type=float, default=0, metavar='WDSGD',
                        help='weight decay (L2 penalty) for SGD (default: 0)')
    parser.add_argument('--lr_Adam', type=float, default=0.001, metavar='LRAdam',
                        help='learning rate for SGD (default: 0.001)')
    parser.add_argument('--betas_Adam', type=tupletype, default=(0.9, 0.999), metavar='betasAdam',
                        help='betas for Adam (default: (0.9, 0.999))')
    parser.add_argument('--weightdecay_Adam', type=float, default=0, metavar='WDAdam',
                        help='weight decay (L2 penalty) for Adam (default: 0)')
    parser.add_argument('--lr_Adagrad', type=float, default=0.01, metavar='LRAdagrad',
                        help='learning rate for Adagrad (default: 0.01)')
    parser.add_argument('--lr_decay_Adagrad', type=float, default=0, metavar='LRDAdagrad',
                        help='learning rate decay for Adagrad (default: 0)')
    parser.add_argument('--weightdecay_Adagrad', type=float, default=0, metavar='WDAdagrad',
                        help='weight decay (L2 penalty) for Adagrad (default: 0)')


    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')

    parser.add_argument("--enable_board", action='store_true', default=True,
                        help="use tensorboard for visualization")
    #parser.add_argument("--board_comment", type=str, default='', help="comment that will be add to the folder name")
    #parser.add_argument("--board_num_samples", type=int, default=8,
    #                    help='number of samples for visualization (default: 8)')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder and save training information
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)
    info_dict = {
        'encoder' : args.encoder,
        'classifier': [25088, 320, 50, 20],
        'optimizer': args.optimizer,
        'dropout': 0.5,
        'activations': 'ReLU',
        'epochs': args.epochs,
        'finetune': args.finetune
    }
    if info_dict['optimizer'] == 'SGD':
        info_dict['lr_SGD'] = args.lr_SGD
        info_dict['momentum_SGD'] = args.momentum_SGD
        info_dict['weightdecay_SGD'] = args.weightdecay_SGD
    elif info_dict['optimizer'] == 'Adam':
        info_dict['lr_Adam'] = args.lr_Adam
        info_dict['betas_Adam'] = args.betas_Adam
        info_dict['weightdecay_Adam'] = args.weightdecay_Adam
    elif info_dict['optimizer'] == 'Adagrad':
        info_dict['lr_Adagrad'] = args.lr_Adagrad
        info_dict['lr_decay_Adagrad'] = args.lr_decay_Adagrad
        info_dict['weightdecay_Adagrad'] = args.weightdecay_Adagrad
    model_folder = args.experiment + '/' + info_dict['encoder'] + '_' +\
        str(info_dict['optimizer']) + '_' + str(info_dict['epochs']) + '_' + str(int(random() * 1e3))
    os.makedirs(model_folder)
    with open(model_folder + '/info.json', 'w') as p:
        json.dump(info_dict, p)


    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    if info_dict['encoder'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
    elif info_dict['encoder'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
    elif info_dict['encoder'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    elif info_dict['encoder'] == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        in_features = model.fc.in_features
    
    if info_dict['finetune'] == False:
        for param in model.parameters():
            param.requires_grad = False
    classifier = Classifier(in_features)
    
    if info_dict['encoder'] == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier
    print(model)
    
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')


    if info_dict['optimizer'] == 'SGD':
        if args.finetune:
            optimizer = optim.SGD(model.parameters(), lr=args.lr_SGD, momentum=args.momentum_SGD,
                weight_decay=args.weightdecay_SGD)
        else:
            if info_dict['encoder'] == 'vgg16':
                optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr_SGD, momentum=args.momentum_SGD,
                    weight_decay=args.weightdecay_SGD)
            else:
                optimizer = optim.SGD(model.fc.parameters(), lr=args.lr_SGD, momentum=args.momentum_SGD,
                    weight_decay=args.weightdecay_SGD)                
    elif info_dict['optimizer'] == 'Adam':
        if args.finetune:
            optimizer = optim.Adam(model.parameters(), lr=args.lr_Adam, betas=args.betas_Adam,
                weight_decay=args.weightdecay_Adam)
        else:
            if info_dict['encoder'] == 'vgg16':
                optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr_Adam, betas=args.betas_Adam,
                    weight_decay=args.weightdecay_Adam)
            else:
                optimizer = optim.Adam(model.fc.parameters(), lr=args.lr_Adam, betas=args.betas_Adam,
                    weight_decay=args.weightdecay_Adam)
    elif info_dict['optimizer'] == 'Adagrad':
        if args.finetune:
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr_Adagrad, lr_decay=args.lr_decay_Adagrad,
                weight_decay=args.weightdecay_Adagrad)
        else:
            if info_dict['encoder'] == 'vgg16':
                optimizer = optim.Adagrad(model.classifier.parameters(), lr=args.lr_Adagrad, lr_decay=args.lr_decay_Adagrad,
                    weight_decay=args.weightdecay_Adagrad)
            else:
                optimizer = optim.Adagrad(model.fc.parameters(), lr=args.lr_Adagrad, lr_decay=args.lr_decay_Adagrad,
                    weight_decay=args.weightdecay_Adagrad)               

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                            transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                            transform=data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Set up of the tensorboard visualization
    writer = SummaryWriter() if args.enable_board else None
    if writer is not None:
        writer.add_text("Parameters", str(info_dict))

    # Training loop
    itrs_train = 0
    itrs_val = 0
    num_val = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation()
        model_file = model_folder + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
