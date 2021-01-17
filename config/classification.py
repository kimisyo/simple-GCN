model_params = {
    'task': 'classification',
    'random_seed': 42,
    'num_epochs': 5,
    'batch_size': 50,
    'file_type': 'sdf',
    'train_data_file': 'input/bace.sdf',
    'label_cols': ['Class'],
    'smiles_col': 'smiles',
    'use_gpu': False,
    'atom_features_size': 62,
    'fingerprints_size': 50,
    'mlp_layer_sizes': [100, 1],
    'conv_layer_sizes': [20, 20, 20],
    'lr': 0.01,
    'hyper':
        {
            'trials': 10,
            'metrics': 'acc',
            'minimize': False,
            'parameters':
                [
                    {'name': 'conv_layer_width', 'type': 'range', 'bounds': [1, 4], 'value_type': 'int'},
                    {'name': 'conv_layer_size', 'type': 'range', 'bounds': [5, 100], 'value_type': 'int'},
                    {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.1], 'value_type': 'float'}
                ]
        }
}


