# simple-GCN

This program is a simple implementation of the of the paper below

["Convolutional Networks on Graphs for Learning Molecular Fingerprints."](https://arxiv.org/pdf/1509.09292.pdf)


# required
- PyTorch 1.7.0
- RDKit 2020.09.2
- scikit-learn 0.23.2
- pytorch-lightning  1.1.4
- ax-platform  0.1.19

# examples
## command line option
```
usage: run.py [-h] -config CONFIG [-hyper] [-es]

optional arguments:
  -h, --help      show this help message and exit
  -config CONFIG  specify config file
  -hyper          run for hyper parameter tuning
  -es             do early stopping
```

## config file
```regression.py
model_params = {
    'task': 'regression', # 'regression' or 'classification'
    'random_seed': 42,
    'num_epochs': 30,
    'batch_size': 80,
    'file_type': 'smi', # 'sdf' (SDF) or 'smi' (SMILES)
    'train_data_file': 'input/Lipophilicity.csv', # input file path
    'label_cols': ['exp'], # target parameter label
    'smiles_col': 'smiles', # smiles column (required when 'file_type' is 'smi')
    'use_gpu': True, # use gpu or not
    'atom_features_size': 62, # fixed value
    'conv_layer_sizes': [20,20,20],  # convolution layer sizes
    'fingerprints_size': 50, # finger print size
    'mlp_layer_sizes': [100, 1], # multi layer perceptron sizes
    'lr': 0.01, #learning late
    'metrics': 'r2', # the metrics for 'check_point' , 'early_stopping', 'hyper'
    'minimize': False, # True if you want to minimize the 'metrics'
    'check_point':
        {
        "dirpath": 'model', # model save path
        "save_top_k": 3, # save top k metrics model
        },
    'early_stopping':
        {
        "min_delta": 0.00,
        "patience": 5,
        "verbose": True,
    },
    'hyper':
        {
            'trials': 10,
            'parameters':
                [
                    {'name': 'batch_size', 'type': 'range', 'bounds': [50, 300], 'value_type': 'int'},
                    {'name': 'conv_layer_width', 'type': 'range', 'bounds': [1, 4], 'value_type': 'int'},
                    {'name': 'conv_layer_size', 'type': 'range', 'bounds': [5, 100], 'value_type': 'int'},
                    {'name': 'fingerprint_size', 'type': 'range', 'bounds': [30, 100], 'value_type': 'int'},
                    {'name': 'mlp_layer_size', 'type': 'range', 'bounds': [30, 100], 'value_type': 'int'},
                    {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.1], 'value_type': 'float'},
                ]
        }
}
```

The model is saved with the contents specified by "check_point".

You can specify the save location with "dirpath" and how many items with the best verification accuracy are saved with "save_top_k".

The value of the "mertics" is used at that time, and whether the metircs should be large or small is specified by "minimize".

This metrics is common to the following early stopping and hyper parameter search indexes.

## train model

```
$ python run.py -config config/regression.py
Global seed set to 42
GPU available: True, used: True
TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                     | Type       | Params
--------------------------------------------------------
0 | activation               | ReLU       | 0
1 | graph_conv_layer_list    | ModuleList | 16.6 K
2 | graph_pooling_layer_list | ModuleList | 3.1 K
3 | mlp_layer_list           | ModuleList | 5.2 K
4 | batch_norm_list          | ModuleList | 200
--------------------------------------------------------
25.2 K    Trainable params
0         Non-trainable params
25.2 K    Total params
Epoch 2: 100%|████████████████████████████| 210/210 [00:12<00:00, 17.14it/s, loss=0.856, v_num=99]
train loss=0.8300003409385681
train r2=0.27188658714294434
validation loss=1.0553514957427979
validation r2=0.33314740657806396

```

## hyper parameter tuning
If you want to perform hyper parameter search, specify the "-hyper" option in the argument of "run.py",
and set the number of trials and search range to "hyper" in the config file.

See the ax-platform documentation for how to specify the search range.

```sh
$ python run.py -config config/regression.py -hyper
```

If hyper parameter search is not specified, the parameters specified directly under "model_params" will be used.

In the case of hyper parameter search, the model is not saved.

### tuning parameter you can specify

- batch size
- graph convoluation layer sidth (depth)
- graph convoluation layer size
- fingerprint size
- mlp_layer_size
- learning rate

## early stopping
If you specify the es option, early stopping is performed on each attempt.

You can set the stopping conditions with "early_stopping" in the config file.

Please see here for details.

https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.EarlyStopping.html


## TensorBoard visualization
A folder "lightning_logs" is created directly under the execution directory.

You can visualize the learning status by specifying the folder during learning and starting the Tensor Board.

Install Tensor Board and run the following from another terminal while learning.

```sh
tensorboard --logdir lightning_logs /
```

If you hit http: // localhost: 6006 / in your browser, you can see the learning status.


