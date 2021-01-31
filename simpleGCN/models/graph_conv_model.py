import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy, auroc
from simpleGCN.layers.graph_conv_layer import GraphConvLayer, GraphPoolingLayer


#class GraphConvModel(nn.Module):
class GraphConvModel(pl.LightningModule):

    def __init__(self,
                 device_ext=torch.device('cpu'),
                 task="regression",
                 atom_features_size=62,
                 conv_layer_sizes=[20, 20, 20],
                 fingerprints_size=50,
                 mlp_layer_sizes=[100, 1],
                 activation=nn.ReLU(),
                 normalize=True,
                 lr=0.01,
                 ):

        super().__init__()

        self.device_ext = device_ext
        self.task = task
        self.activation = activation
        self.normalize = normalize
        self.number_atom_features = atom_features_size
        self.graph_conv_layer_list = nn.ModuleList()
        self.graph_pooling_layer_list = nn.ModuleList()
        self.mlp_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.lr = lr

        prev_layer_size = atom_features_size

        for i, layer_size in enumerate(conv_layer_sizes):
            self.graph_conv_layer_list.append(GraphConvLayer(prev_layer_size, layer_size))
            self.graph_pooling_layer_list.append(GraphPoolingLayer(layer_size, fingerprints_size))
            prev_layer_size = layer_size

        # for model in self.graph_conv_layer_list:
        #     for param in model.parameters():
        #         param.requires_grad = False;
        #
        # for model in self.graph_pooling_layer_list:
        #     for param in model.parameters():
        #         param.requires_grad = False;


        prev_layer_size = fingerprints_size
        for i, layer_size in enumerate(mlp_layer_sizes):
            self.mlp_layer_list.append(torch.nn.Linear(prev_layer_size, layer_size, bias=True))
            prev_layer_size = layer_size

        if normalize:
            for i, mlp_layer in enumerate(self.mlp_layer_list):
                if i < len(self.mlp_layer_list) -1 :
                    self.batch_norm_list.append(torch.nn.BatchNorm1d(mlp_layer.out_features))

    def forward(self, array_rep, atom_features, bond_features):

        all_layer_fps = []
        for graph_conv_layer, graph_pooling_layer in zip(self.graph_conv_layer_list, self.graph_pooling_layer_list):
            atom_features = graph_conv_layer(array_rep, atom_features, bond_features)
            fingerprint = graph_pooling_layer(array_rep, atom_features)
            all_layer_fps.append(torch.unsqueeze(fingerprint, dim=0))

        layer_output = torch.cat(all_layer_fps, axis=0)
        layer_output = torch.sum(layer_output, axis=0)

        # MLP Layer
        x = layer_output.float()
        for i, mlp_layer in enumerate(self.mlp_layer_list):
            x = mlp_layer(x)
            if i < len(self.mlp_layer_list) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.normalize:
                    x = self.batch_norm_list[i](x)

        return x

    def training_step(self, batch, batch_idx):

        array_rep, labels_list = batch
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']

        atom_features = torch.tensor(atom_features, dtype=torch.float)
        bond_features = torch.tensor(bond_features, dtype=torch.float)

        if self.task == "regression":
            labels = torch.tensor(labels_list, dtype=torch.float)
        else:
            labels = torch.tensor(labels_list, dtype=torch.float)

        atom_features = atom_features.to(self.device_ext)
        bond_features = bond_features.to(self.device_ext)
        labels = labels.to(self.device_ext)

        y_pred = self(array_rep, atom_features, bond_features)

        if self.task == "regression":
            loss = F.mse_loss(y_pred, labels)
            # https://github.com/pytorch/ignite/issues/453
            var_y = torch.var(labels, unbiased=False)
            r2 = 1.0 - F.mse_loss(y_pred, labels, reduction="mean") / var_y
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)

            y_pred_proba = torch.sigmoid(y_pred)
            y_pred = y_pred_proba > 0.5
            acc = accuracy(y_pred, labels)
            y_pred_proba = y_pred_proba.to('cpu').detach().numpy().tolist()
            from sklearn import metrics
            try:
                rocauc = metrics.roc_auc_score(labels_list, y_pred_proba)
            except Exception as es:
                rocauc = 0

        if self.task == "regression":
            ret = {'loss': loss, 'train_r2': r2}
        else:
            ret = {'loss': loss, 'train_acc': acc, 'train_rocauc': torch.tensor(rocauc, dtype=torch.float)}

        return ret

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss)

        if self.task == "regression":
            r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
            self.log('train_r2', r2)
        else:
            acc = torch.stack([x['train_acc'] for x in outputs]).mean()
            self.log('train_acc', acc)
            rocauc = torch.stack([x['train_rocauc'] for x in outputs]).mean()
            self.log('train_rocauc', rocauc)

    def validation_step(self, batch, batch_idx):

        array_rep, labels_list = batch
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']

        atom_features = torch.tensor(atom_features, dtype=torch.float)
        bond_features = torch.tensor(bond_features, dtype=torch.float)
        if self.task == "regression":
            labels = torch.tensor(labels_list, dtype=torch.float)
        else:
            labels = torch.tensor(labels_list, dtype=torch.float)

        atom_features = atom_features.to(self.device)
        bond_features = bond_features.to(self.device)
        labels = labels.to(self.device)

        y_pred = self(array_rep, atom_features, bond_features)

        if self.task == "regression":
            loss = F.mse_loss(y_pred, labels)
            # https://github.com/pytorch/ignite/issues/453
            var_y = torch.var(labels, unbiased=False)
            r2 = 1.0 - F.mse_loss(y_pred, labels, reduction="mean") / var_y
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            y_pred_proba = torch.sigmoid(y_pred)
            y_pred = y_pred_proba > 0.5
            acc = accuracy(y_pred, labels)
            y_pred_proba = y_pred_proba.to('cpu').detach().numpy().tolist()
            from sklearn import metrics
            try:
                rocauc = metrics.roc_auc_score(labels_list, y_pred_proba)
            except Exception as es:
                rocauc = 0

        if self.task == "regression":
            ret = {'val_loss': loss, 'val_r2': r2}
        else:
            ret = {'val_loss': loss, 'val_acc': acc, 'val_rocauc': torch.tensor(rocauc, dtype=torch.float)}

        return ret

    def validation_epoch_end(self, outputs):

        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss)

        if self.task == "regression":
            r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
            self.log('val_r2', r2)
        else:
            acc = torch.stack([x['val_acc'] for x in outputs]).mean()
            self.log('val_acc', acc)
            rocauc = torch.stack([x['val_rocauc'] for x in outputs]).mean()
            self.log('val_rocauc', rocauc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
