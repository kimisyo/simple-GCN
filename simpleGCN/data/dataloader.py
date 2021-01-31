import pandas as pd


class SMILESDataLoader:

    def __init__(self, csv_path, smiles_col="smiles", label_props=["label"]):
        # read csv file
        df = pd.read_csv(csv_path)

        mols = []
        labels_list = []
        for i, samples in enumerate(df[smiles_col].values):
            from rdkit import Chem
            mol = Chem.MolFromSmiles(samples)
            if mol is not None:
                mols.append(mol)
                labels = [None] * len(label_props)
                for j, label in enumerate(label_props):
                    labels[j] = df[label].values[i]

                labels_list.append(labels)

        self.mols = mols
        self.labels_list = labels_list


class SDFDataLoader:

    def __init__(self, sdf_path, label_props=["label"]):

        from rdkit import Chem
        sdf_sup = Chem.SDMolSupplier(sdf_path)

        mols = []
        labels_list = []
        for i, mol in enumerate(sdf_sup):
            if not mol:
                continue

            mols.append(mol)

            labels = [None] * len(label_props)
            for j, label in enumerate(label_props):
                try:
                    labels[j] = float(mol.GetProp(label))
                except Exception as ex:
                    labels[j] = None

            labels_list.append(labels)

        self.mols = mols
        self.labels_list = labels_list

def main():
    #loader = SMILESDataLoader("input/BBBP.csv", label_cols=["p_np"])
    #print(len(loader.mols))
    #print(len(loader.labels_list))

    loader = SDFDataLoader("input/bace.sdf", label_props=["Class"])
    print(len(loader.mols))
    print(loader.labels_list)

if __name__ == "__main__":
    main()