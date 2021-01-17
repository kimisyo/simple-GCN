import numpy as np
from torch.utils import data
from simpleGCN.data.dataloader import SMILESDataLoader, SDFDataLoader
from simpleGCN.feat.mol_graph import graph_from_mol_tuple, degrees


class MoleculeDataset(data.Dataset):

    def __init__(self, mol_list, label_list):
        self.mol_list = mol_list
        self.label_list = label_list

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        return self.mol_list[index], self.label_list[index]


class AtomDataset(data.Dataset):

    def __init__(self, mol_list, label_list, atom_label_names=None, atom_feature_names=None):

        self.atom_label_names = atom_label_names
        self.atom_feature_names = atom_feature_names

        self.mol_list = mol_list
        self.label_list = label_list

        self.atom_label_list = []
        self.atom_feature_list = []

        for mol in mol_list:
            atom_label_values_list = [None] * len(mol.GetAtoms())
            atom_feature_values_list = [None] * len(mol.GetAtoms())

            for atom in mol.GetAtoms():
                if atom_label_names:
                    atom_label_values = []
                    for i, atom_label_name in enumerate(atom_label_names):
                        try:
                            atom_label_values.append(atom.GetDoubleProp(atom_label_name))
                        except Exception as ex:
                            atom_label_values.append(None)
                    atom_label_values_list[atom.GetIdx()] = atom_label_values

                if atom_feature_names:
                    atom_feature_values = []
                    for i, atom_feature_name in enumerate(atom_feature_names):
                        try:
                            atom_feature_values.append(atom.GetDoubleProp(atom_feature_name))
                        except Exception as ex:
                            atom_feature_values.append(None)

                    atom_feature_values_list[atom.GetIdx()] = atom_feature_values

            self.atom_label_list.append(atom_label_values_list)
            self.atom_feature_list.append(atom_feature_values_list)


    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, index):
        if self.atom_label_names:
            if self.atom_feature_names:
                return self.mol_list[index], self.label_list[index], self.atom_label_list[index], self.atom_feature_list[index]
            else:
                return self.mol_list[index], self.label_list[index], self.atom_label_list[index]
        else:
            return self.mol_list[index], self.label_list[index]


def gcn_collate_fn(batch):

    mols = []
    labels = []
    atom_labels = []
    atom_features = []

    for i, (mol, label, *atom_data) in enumerate(batch):
        mols.append(mol)
        labels.append(label)
        if len(atom_data) > 0:
            atom_labels.append(atom_data[0])
            if len(atom_data) > 1:
                atom_features.append(atom_data[1])

    molgraph = graph_from_mol_tuple(mols, atom_labels, atom_features)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix': molgraph.rdkit_ix_array(),
                'atom_labels': molgraph.labels_array()
                }  # For plotting only.

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    #print(arrayrep)
    return arrayrep, labels


def main():
    #loader = SMILESDataLoader("input/BBBP.csv", label_cols=["p_np"])
    #print(len(loader.mols))
    #print(len(loader.labels_list))

    # loader = SDFDataLoader("input/bace.sdf", label_props=["Class"])
    #md = MoleculeDataset(loader.mols, loader.labels_list)
    #dataloader = data.DataLoader(md, batch_size=1, shuffle=False, collate_fn=gcn_collate_fn)
    #array_rep, labels = next(iter(dataloader))
    #print(array_rep["atom_list"])

    loader = SDFDataLoader("input/atom.sdf", label_props=["PRIMARY_SOM_1A2"])
    md = AtomDataset(loader.mols, loader.labels_list, atom_label_names=["relSPAN4"])
    print(md.__getitem__(0))
    dataloader = data.DataLoader(md, batch_size=2, shuffle=False, collate_fn=gcn_collate_fn)
    array_rep, labels = next(iter(dataloader))
    print(array_rep["atom_labels"])
    #print(labels)

if __name__ == "__main__":
    main()