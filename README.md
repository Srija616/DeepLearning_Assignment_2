# CS6910-DeepLearning_Assignment 2

Wandb Report Link: https://wandb.ai/srija17199/DL-Assignment2/reports/CS6910-Assignment-2-Srija-Anand--Vmlldzo3NDM5ODcy?accessToken=itkusfx8r8j6f7vzpxmvgot2hon7zt9yquriz231qiuy7k9m9xg9h5gb12pv739u

## Part A
First install the conda environment (a2_env.yaml) using Python 3.9
To run the code:
To run training or test for one configuration, you can run the models/model.py. Below is its argparser along with the default data. You need to download the iNaturalist data and pass the train data path and test data path in the parser.

```python
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
    parser.add_argument("--max_epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_arg", type=bool, default=False, help="Data Augmentation")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, help="Path to the test dataset")
```
You can also run the code for the same using the Python Notebook: PartA_Q2.ipynb

## Part B
You can run part B directly from the Python Notebook PartB_Q1.ipynb. 
The finetuning model class is defined in models/model_partb.py
