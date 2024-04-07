import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import numpy as np
from torch.utils.data import DataLoader,Dataset
import argparse

class ConvBlocks(nn.Module):
    ''' Defines 5 convolution layers used in a CNN
    '''
    def __init__(self,in_channels,num_filters,filter_size,activation, batch_norm = False, filter_org = 1):
        super().__init__()
        
        self.activation = activation
        self.batch_norm=batch_norm
        self.filter_org = filter_org
        self.filter_size = filter_size
        in_channels_list =[in_channels]
        out_channels_list = [num_filters]

        for i in range (4):
            in_channels_list.append(out_channels_list[-1])
            out_channels_list.append(int(filter_org*out_channels_list[-1]))

        # ModuleList is used to store all the five layers of Conv2D, BatchNormalization and max pool
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range (5):
            self.layers.append(nn.Conv2d(in_channels=in_channels_list[i],out_channels=out_channels_list[i],kernel_size=self.filter_size,stride=(1, 1),padding=(1, 1),bias=False))
            self.bn_layers.append(nn.BatchNorm2d(out_channels_list[i]))
        self.pool  = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x):

        if self.batch_norm == False:
            for i in range (5):
                x = self.pool(self.activation(self.layers[i](x)))
        else:
            for i in range (5):
                x = self.pool(self.activation(self.bn_layers[i](self.layers[i](x))))
        return x


class Model(pl.LightningModule):
    ''' CNN Network - 5 Conv layers + activation (relu) + max-pool
    '''
    def __init__(self, 
                 in_channels, 
                 num_filters, 
                 filter_size, 
                 activation, 
                 neurons_dense, 
                 image_shape, 
                 batch_norm, 
                 filter_org,
                 classes = 10,
                 dropout = 0.0,
                 batch_size = 32,
                 lr = 1e-3,
                 train_dataset=None, 
                 val_dataset=None, 
                 test_dataset=None):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.accuracy = Accuracy(task='multiclass', num_classes=classes)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.lr = lr
        self.best_val_acc = 0
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()
        
        self.conv_blocks = ConvBlocks(in_channels, num_filters, filter_size, self.activation, batch_norm, filter_org)
        
        fc1_in_channels = self.get_output_shape(image_shape)
        self.fc1 = nn.Linear(fc1_in_channels,neurons_dense, bias=True)  
        self.output = nn.Linear(neurons_dense, 10, bias=True)   
    
    def get_output_shape(self, image_shape):
        return self.conv_blocks(torch.zeros(*(image_shape))).numel()

    def forward(self, x):
        x = self.conv_blocks(x) 
        x = self.activation(self.dropout(self.fc1(x.reshape(x.shape[0],-1))))
        x = F.softmax(self.output(x),dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"loss": loss, "train_acc": acc}
        self.log_dict(metrics, on_epoch=True)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
        metrics = {"val_loss": loss, "val_acc": acc, "best_val_acc": self.best_val_acc}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics)
        return metrics

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, shuffle=False, num_workers=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for your model")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--num_filters", type=int, default=32, help="Number of filters in the convolutional layers")
    parser.add_argument("--filter_size", type=int, default=5, help="Size of the filters in the convolutional layers")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function for the model")
    parser.add_argument("--neurons_dense", type=int, default=32, help="Number of neurons in the dense layer")
    parser.add_argument("--image_shape", type=tuple, default=(3, 100, 100), help="Shape of the input images")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Whether to use batch normalization")
    parser.add_argument("--filter_org", type=float, default=0.5, help="Filter organization parameter")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes in the classification task")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # Add arguments for dataset paths if required
    # parser.add_argument("--train_dataset", type=str, help="Path to the training dataset")
    # parser.add_argument("--val_dataset", type=str, help="Path to the validation dataset")
    # parser.add_argument("--test_dataset", type=str, help="Path to the test dataset")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # Instantiate Model with argparse values
    model = Model(args.in_channels, args.num_filters, args.filter_size, args.activation, args.neurons_dense, args.image_shape, args.batch_norm, args.filter_org, args.classes, args.dropout, args.batch_size, args.lr, args.train_dataset, args.val_dataset, args.test_dataset)
    print(model)
    
    # sample_input = torch.randn(16,3,100,100)
    # print('INPUT: ', sample_input.shape)
    # output = model(sample_input)
    # print('OUTPUT: ', output.shape)

