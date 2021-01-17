model_params = {
    'task': 'regression',
    'random_seed': 42,
    'num_epochs': 2,
    'batch_size': 20,
    'file_type': 'smi',
    'train_data_file': 'input/Lipophilicity.csv',
    'label_cols': ['exp'],
    'smiles_col': 'smiles',
    'use_gpu': True,
    'atom_features_size': 62,
    'fingerprints_size': 50,
    'mlp_layer_sizes': [100, 1],
    'conv_layer_sizes': [20, 20, 20],
    'lr': 0.01,
    'hyper':
        {
            'trials': 10,
            'metrics': 'r2',
            'minimize': False,
            'parameters':
                [
                    {'name': 'conv_layer_width', 'type': 'range', 'bounds': [1, 4], 'value_type': 'int'},
                    {'name': 'conv_layer_size', 'type': 'range', 'bounds': [5, 100], 'value_type': 'int'},
                    {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.1], 'value_type': 'float'}
                ]
        }
}


