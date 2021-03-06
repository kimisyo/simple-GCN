model_params = {
    'task': 'classification', # 'regression' or 'classification'
    'random_seed': 42,
    'num_epochs': 100,
    'batch_size': 50,
    'file_type': 'Smi', # 'sdf' (SDF) or 'smi' (SMILES)
     #'train_data_file': 'input/bace.sdf', # input file path
     'train_data_file': 'input/BBBP.csv',
    #'label_cols': ['Class'], # target parameter label
    'label_cols': ['p_np'], # target parameter label
    'smiles_col': 'smiles', # smiles column (required when 'file_type' is 'smi')
    'use_gpu': True, # use gpu or not
    'atom_features_size': 62, # fixed value
    'conv_layer_sizes': [20, 20, 20],  # convolution layer sizes
    'fingerprints_size': 50, # finger print size
    'mlp_layer_sizes': [100, 1], # multi layer perceptron sizes
    'lr': 0.01, #learning late
    'metrics': 'rocauc', # the metrics for 'check_point' , 'early_stopping', 'hyper'
    'minimize': False, # True if you want to minimize the 'metrics'
    'check_point':
        {
        "dirpath": 'model', # model save path
        "save_top_k": 3, # save top k metrics model
        },
    'early_stopping': # see https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.EarlyStopping.html
        {
        "min_delta": 0.00,
        "patience": 3,
        "verbose": True,
    },
    'hyper':
        {
            'trials': 20,
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


