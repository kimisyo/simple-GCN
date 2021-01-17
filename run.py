import argparse
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
import ax

from simpleGCN.data.dataloader import SMILESDataLoader, SDFDataLoader
from simpleGCN.data.dataset import AtomDataset, MoleculeDataset, gcn_collate_fn
from simpleGCN.models.graph_conv_model import GraphConvModel


def print_result(trainer):
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

    #パラメータの取得
    conv_layer_width = parameters["conv_layer_width"]
    conv_layer_size = parameters["conv_layer_size"]
    lr = parameters["lr"]

    print("conv_layer_width={0}".format(conv_layer_width))
    print("conv_layer_size={0}".format(conv_layer_size))

    conv_layer_sizes = []
    for i in range(conv_layer_width):
        conv_layer_sizes.append(conv_layer_size)

    model = GraphConvModel(
        device_ext=params["device"],
        task=params["task"],
        conv_layer_sizes=conv_layer_sizes,
        fingerprints_size=params["fingerprints_size"],
        lr=lr
    )

    trainer = pl.Trainer(
        max_epochs=params["num_epochs"],
        gpus=params["gpu"],
    )
    trainer.fit(model, params["data_loader_train"], params["data_loader_val"])
    print_result(trainer)

    key = "val_{0}".format(params["hyper"]["metrics"])
    return trainer.callback_metrics[key].item()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-mode", type=str, default="train")
    parser.add_argument("-hyper", action='store_true')
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
    data_loader_train = data.DataLoader(molecule_dataset_train, batch_size=params["batch_size"], shuffle=False, collate_fn=gcn_collate_fn)

    molecule_dataset_test = MoleculeDataset(X_test, y_val)
    data_loader_val = data.DataLoader(molecule_dataset_test, batch_size=params["batch_size"], shuffle=False, collate_fn=gcn_collate_fn)

    if params["use_gpu"] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu = 1
    else:
        device = torch.device('cpu')
        gpu = 0

    if args.hyper:
        params["data_loader_train"] = data_loader_train
        params["data_loader_val"] = data_loader_val
        params["device"] = device
        params["gpu"] = gpu

        results = ax.optimize(params["hyper"]["parameters"],
                              evaluation_function,
                              random_seed=params["random_seed"],
                              minimize=params["hyper"]["minimize"],
                              total_trials=params["hyper"]["trials"]
                              )
        print(results)
    else:
        model = GraphConvModel(
         device_ext=device,
         task=params["task"],
         conv_layer_sizes=params["conv_layer_sizes"],
         fingerprints_size=params["fingerprints_size"],
         lr=params["lr"]
        )
        trainer = pl.Trainer(
             max_epochs=params["num_epochs"],
             gpus=gpu
        )
        trainer.fit(model, data_loader_train, data_loader_val)
        print_result(trainer)

if __name__ == "__main__":
    main()
