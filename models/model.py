import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import numpy as np

class ConvBlocks(nn.Module):
    ''' Defines 5 convolution layers used in a CNN
    '''
    def __init__(self,in_channels,num_filters,filter_size,activation, batch_norm = False, filter_org = 1):
        super().__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif self.activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()

        self.batch_norm=batch_norm
        self.filter_org = filter_org
        in_channels_list =[in_channels]
        out_channels_list = [num_filters]

        for i in range (4):
            in_channels_list.append(out_channels_list[-1])
            out_channels_list.append(int(filter_org*out_channels_list[-1]))

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range (5):
            self.layers.append(nn.Conv2d(in_channels=in_channels_list[i],out_channels=out_channels_list[i],kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False))
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
                 train_dataset=None, 
                 val_dataset=None, 
                 test_dataset=None):
        
        super().__init__()

        self.accuracy = Accuracy(task='multiclass', num_classes=classes)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif self.activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()
        
        self.conv_blocks = ConvBlocks(in_channels, num_filters, filter_size, activation, batch_norm, filter_org)
        
        fc1_in_channels = self.get_output_shape(image_shape)
        self.fc1 = nn.Linear(fc1_in_channels,neurons_dense, bias=True)  
        self.output = nn.Linear(neurons_dense, 10, bias=True)   
    
    def get_output_shape(self, image_shape):
        return self.conv_blocks(torch.zeros(*(image_shape))).numel()

    def forward(self, x):
        x = self.conv_blocks(x) 
        x = self.activation(self.fc1(x.reshape(x.shape[0],-1)))
        x = F.softmax(self.output(x),dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"loss": loss, "train_acc": acc}
        self.log_dict(metrics)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"loss": loss, "val_acc": acc}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"loss": loss, "val_acc": acc}
        self.log_dict(metrics)
        return metrics

if __name__ == '__main__':
    # Sample Hyperparameters
    # initialize the activation here
    activation = 'relu'
    num_filters = 32
    filter_size = 5
    in_channels = 3
    neurons_dense = 32
    filter_org = 0.5
    batch_norm = True
    batch_size = 16

    # Instantiate Model
    model = Model(in_channels,num_filters,filter_size,activation,neurons_dense, (1, 3, 100, 100), batch_size, batch_norm, filter_org)
    print (model)

    sample_input = torch.randn(16,3,100,100)
    print('INPUT: ', sample_input.shape)
    output = model(sample_input)
    print('OUTPUT: ', output.shape)
