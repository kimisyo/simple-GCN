import argparse
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import ax
from pytorch_lightning.callbacks import ModelCheckpoint

from simpleGCN.data.dataloader import SMILESDataLoader, SDFDataLoader
from simpleGCN.data.dataset import AtomDataset, MoleculeDataset, gcn_collate_fn
from simpleGCN.models.graph_conv_model import GraphConvModel


def print_result(trainer):
    # print(trainer.callback_metrics)
    print("train loss={0}".format(trainer.callback_metrics['train_loss']))
    if params["task"] == "regression":
        print("train r2={0}".format(trainer.callback_metrics['train_r2'].item()))
    else:
        print("train acc={0}".format(trainer.callback_metrics['train_acc'].item()))
        print("train rocauc={0}".format(trainer.callback_metrics['train_rocauc'].item()))

    print("validation loss={0}".format(trainer.callback_metrics['val_loss']))
    if params["task"] == "regression":
        print("validation r2={0}".format(trainer.callback_metrics['val_r2'].item()))
    else:
        print("validation acc={0}".format(trainer.callback_metrics['val_acc'].item()))
        print("validation rocauc={0}".format(trainer.callback_metrics['val_rocauc'].item()))


def evaluation_function(parameters):

    # reading hyper parameters
    batch_size = parameters["batch_size"]
    conv_layer_width = parameters["conv_layer_width"]
    conv_layer_size = parameters["conv_layer_size"]
    fingerprint_size = parameters["fingerprint_size"]
    mlp_layer_size = parameters["mlp_layer_size"]
    lr = parameters["lr"]

    data_loader_train = data.DataLoader(params["molecule_dataset_train"], batch_size=batch_size, shuffle=False,
                                        collate_fn=gcn_collate_fn)

    data_loader_val = data.DataLoader(params["molecule_dataset_test"], batch_size=batch_size, shuffle=False,
                                      collate_fn=gcn_collate_fn)

    print("batch_size={0}".format(batch_size))
    print("conv_layer_width={0}".format(conv_layer_width))
    print("conv_layer_size={0}".format(conv_layer_size))
    print("fingerprint_size={0}".format(fingerprint_size))
    print("mlp_layer_size={0}".format(mlp_layer_size))

    conv_layer_sizes = []
    for i in range(conv_layer_width):
        conv_layer_sizes.append(conv_layer_size)

    model = GraphConvModel(
        device_ext=params["device"],
        task=params["task"],
        conv_layer_sizes=conv_layer_sizes,
        fingerprints_size=fingerprint_size,
        mlp_layer_sizes=[mlp_layer_size, 1],
        lr=lr
    )

    callbacks = []
    if params["es"]:
        early_stop_callback = EarlyStopping(
            min_delta=params["early_stopping"]["min_delta"],
            patience=params["early_stopping"]["patience"],
            verbose=params["early_stopping"]["verbose"],
            monitor="val_" + params["metrics"],
            mode='min' if params["minimize"] else 'max'
        )
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=params["num_epochs"],
        gpus=params["gpu"],
        callbacks=callbacks
    )
    trainer.fit(model, data_loader_train, data_loader_val)
    print_result(trainer)

    key = "val_{0}".format(params["metrics"])
    return trainer.callback_metrics[key].item()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-mode", type=str, default="train")
    parser.add_argument("-hyper", action='store_true')
    parser.add_argument("-es", action='store_true')
    args = parser.parse_args()

    global params

    params = runpy.run_path(args.config).get('model_params', None)
    pl.seed_everything(params["random_seed"])

    if params["file_type"] == "sdf":
        loader = SDFDataLoader(params["train_data_file"], label_props=params["label_cols"])
    else:
        loader = SMILESDataLoader(params["train_data_file"], smiles_col=params["smiles_col"], label_props=params["label_cols"])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_val = train_test_split(loader.mols, loader.labels_list, shuffle=True, train_size=0.8, random_state=params["random_seed"])
    molecule_dataset_train = MoleculeDataset(X_train, y_train)
    molecule_dataset_test = MoleculeDataset(X_test, y_val)

    if params["use_gpu"] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu = 1
    else:
        device = torch.device('cpu')
        gpu = 0

    callbacks = [EarlyStopping(monitor='val_loss')]
    if args.hyper:
        params["molecule_dataset_train"] = molecule_dataset_train
        params["molecule_dataset_test"] = molecule_dataset_test
        params["device"] = device
        params["gpu"] = gpu
        params["es"] = True if args.es else False

        results = ax.optimize(params["hyper"]["parameters"],
                              evaluation_function,
                              random_seed=params["random_seed"],
                              minimize=params["minimize"],
                              total_trials=params["hyper"]["trials"]
                              )
        print(results)
    else:
        data_loader_train = data.DataLoader(molecule_dataset_train, batch_size=params["batch_size"], shuffle=False,
                                            collate_fn=gcn_collate_fn)

        data_loader_val = data.DataLoader(molecule_dataset_test, batch_size=params["batch_size"], shuffle=False,
                                          collate_fn=gcn_collate_fn)

        model = GraphConvModel(
         device_ext=device,
         task=params["task"],
         conv_layer_sizes=params["conv_layer_sizes"],
         fingerprints_size=params["fingerprints_size"],
         mlp_layer_sizes=params["mlp_layer_sizes"],
         lr=params["lr"]
        )

        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            monitor='val_' + params["metrics"],
            dirpath=params["check_point"]["dirpath"],
            filename="model-{epoch:02d}-{val_" + params["metrics"]+":.5f}",
            save_top_k=params["check_point"]["save_top_k"],
            mode='min' if params["minimize"] else 'max'
        )
        callbacks.append(checkpoint_callback)

        if args.es:
            early_stop_callback = EarlyStopping(
            min_delta=params["early_stopping"]["min_delta"],
            patience=params["early_stopping"]["patience"],
            verbose=params["early_stopping"]["verbose"],
            monitor="val_"+ params["metrics"],
            mode='min' if params["minimize"] else 'max'
            )
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(
             max_epochs=params["num_epochs"],
             gpus=gpu,
             callbacks=callbacks
        )
        trainer.fit(model, data_loader_train, data_loader_val)
        print_result(trainer)

if __name__ == "__main__":
    main()
