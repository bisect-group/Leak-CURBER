import pandas as pd
import numpy as np
from utils import Seq_to_vec, GetMACCSKeys, get_molT5_embed

# replace the model paths with your local paths
prott5_model = "/your_path/prot_t5_xl_uniref50/"
molt5_model = "/your_path/molt5-base-smiles2caption/"

def get_feats(inp_fpath, out_fpath):
    """
    seq_list: include all protein sequences
    smi_list: include all substrate SMILES 
    out_fpath: path of saved features
    """
    inp_df = pd.read_csv(inp_fpath, index_col=0)
    seq_list = inp_df["Sequence"].values.tolist()
    smi_list = inp_df["Smiles"].values.tolist()
    index = inp_df.index.tolist()

    prot_feats = Seq_to_vec(seq_list, prott5_model)
    sbt_molt5_feats = get_molT5_embed(smi_list, molt5_model)
    sbt_macc = GetMACCSKeys(smi_list)

    merge_feats = np.concatenate([prot_feats, sbt_molt5_feats, sbt_macc], axis=1)
    final_df = pd.DataFrame(merge_feats, index=index)
    final_df.to_pickle(out_fpath)

if __name__ == "__main__":
    # If you want to generate features for enzyme-substrate pairs in the kcat dataset, you can run the following command.
    get_feats("../datasets/kcat-data_0.4simi-10fold.csv", "kcat-datasets_features.pkl")
