import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import pickle
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from pandarallel import pandarallel
from hydra import initialize, compose
from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
smiles_hash = chem_utils.smiles_hash
canonicalize_smiles = chem_utils.canonicalize_smiles
get_uniprot_acc_json = chem_utils.get_uniprot_acc_json
add_uniprot_date_column = chem_utils.add_uniprot_date_column
assign_experimental_and_af_pdbs = chem_utils.assign_experimental_and_af_pdbs


class BindingAffinityDatasetBuilder:
    SUPPORTED_VALUE_TYPES = {"unified", "kd", "ec50", "ic50"}

    def __init__(self, cfg: DictConfig):
        LOG_PATH = Path(cfg.binding_affinity_dataset.log_dir)

        self.PANDARALLEL_NB_WORKERS = min(
            cfg.binding_affinity_dataset.pandarallel.nb_workers, os.cpu_count()
        )
        self.PANDARALLEL_PROGRESS_BAR = (
            cfg.binding_affinity_dataset.pandarallel.progress_bar or False
        )

        self.BINDINGDB_TSV_FILE_PATH = Path(
            cfg.binding_affinity_dataset.bindingdb_tsv_file_path
        )
        self.BINDINGDB_DATA_PATH = Path(
            cfg.binding_affinity_dataset.bindingdb_data_path
        )
        self.BINDING_AFFINITY_SMILES_PICKLE_PATH = Path(
            cfg.binding_affinity_dataset.binding_affinity_smiles_pickle_path
        )
        self.BINDING_AFFINITY_SEQUENCES_PICKLE_PATH = Path(
            cfg.binding_affinity_dataset.binding_affinity_sequences_pickle_path
        )
        self.VALUE_TYPES = list(dict.fromkeys(cfg.binding_affinity_dataset.value_types))
        unsupported_value_types = set(self.VALUE_TYPES) - self.SUPPORTED_VALUE_TYPES
        if unsupported_value_types:
            raise ValueError(
                "Unsupported binding_affinity_dataset.value_types: "
                f"{sorted(unsupported_value_types)}. "
                f"Supported values: {sorted(self.SUPPORTED_VALUE_TYPES)}"
            )
        if not self.VALUE_TYPES:
            raise ValueError("binding_affinity_dataset.value_types must not be empty.")
        self.SAVE_1D = cfg.binding_affinity_dataset.save_1d_dataset
        self.SAVE_3D = cfg.binding_affinity_dataset.save_3d_dataset
        if not (self.SAVE_1D or self.SAVE_3D):
            raise ValueError(
                "At least one of binding_affinity_dataset.save_1d_dataset or "
                "binding_affinity_dataset.save_3d_dataset must be true."
            )

        self.BINDING_AFFINITY_1D_DATASET_PARQUET_FILE_PATH = Path(
            cfg.binding_affinity_dataset.binding_affinity_1d_dataset_parquet_file_path
        )
        self.BINDING_AFFINITY_3D_DATASET_PARQUET_FILE_PATH = Path(
            cfg.binding_affinity_dataset.binding_affinity_3d_dataset_parquet_file_path
        )
        self.VALUE_TYPE_DATASET_PATHS = {
            value_type: {
                "1d": Path(
                    cfg.binding_affinity_dataset.dataset_paths[value_type].parquet_1d
                ),
                "3d": Path(
                    cfg.binding_affinity_dataset.dataset_paths[value_type].parquet_3d
                ),
            }
            for value_type in self.VALUE_TYPES
        }

        for path in [
            LOG_PATH,
            self.BINDINGDB_DATA_PATH,
            self.BINDING_AFFINITY_SMILES_PICKLE_PATH.parent,
            self.BINDING_AFFINITY_SEQUENCES_PICKLE_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        for dataset_paths in self.VALUE_TYPE_DATASET_PATHS.values():
            for path in dataset_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.binding_affinity_dataset.log_file_name
        ).get_logger()

        if not self.BINDINGDB_TSV_FILE_PATH.exists():
            msg = f"BindingDB data file not found at {self.BINDINGDB_TSV_FILE_PATH}. Please download it from https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp. Download the 'All Data' TSV file."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

    def _read_bindingdb_data(self):
        drop_cols = [
            "BindingDB Reactant_set_id",
            "Ligand InChI",
            "Ligand InChI Key",
            "Curation/DataSource",
            "Article DOI",
            "PubChem AID",
            "Patent Number",
            "Authors",
            "Date of publication",
            "Date in BindingDB",
            "Institution",
            "Link to Ligand in BindingDB",
            "Link to Target in BindingDB",
            "Link to Ligand-Target Pair in BindingDB",
            "Ligand HET ID in PDB",
            "DrugBank ID of Ligand",
            "IUPHAR_GRAC ID of Ligand",
        ] + [
            item
            for sublist in [
                [
                    f"BindingDB Target Chain Sequence {i}",
                    f"UniProt (SwissProt) Recommended Name of Target Chain {i}",
                    f"UniProt (SwissProt) Entry Name of Target Chain {i}",
                    f"UniProt (SwissProt) Secondary ID(s) of Target Chain {i}",
                    f"UniProt (SwissProt) Alternative ID(s) of Target Chain {i}",
                    f"UniProt (TrEMBL) Submitted Name of Target Chain {i}",
                    f"UniProt (TrEMBL) Entry Name of Target Chain {i}",
                    f"UniProt (TrEMBL) Primary ID of Target Chain {i}",
                    f"UniProt (TrEMBL) Secondary ID(s) of Target Chain {i}",
                    f"UniProt (TrEMBL) Alternative ID(s) of Target Chain {i}",
                ]
                for i in range(1, 51)
            ]
            for item in sublist
        ]

        self.logger.info("Reading BindingDB columns...")
        all_cols = pd.read_csv(
            self.BINDINGDB_TSV_FILE_PATH, sep="\t", nrows=0
        ).columns.tolist()
        use_cols = [col for col in all_cols if col not in drop_cols]

        self.logger.info(
            f"Reading required {len(use_cols)} out of {len(all_cols)} columns from BindingDB."
        )
        bindingdb_df = pd.read_csv(
            self.BINDINGDB_TSV_FILE_PATH,
            sep="\t",
            usecols=use_cols,
            low_memory=False,
        )
        self.logger.info(f"BindingDB data shape: {bindingdb_df.shape}")
        return bindingdb_df

    def _explode_multichain_with_metadata(self, df, chain_n=50):
        chain_fields = [
            "UniProt (SwissProt) Primary ID of Target Chain",
            "PDB ID(s) of Target Chain",
        ]
        chain_specific_cols = [
            f"{field} {i}" for field in chain_fields for i in range(1, chain_n + 1)
        ]
        meta_cols = [c for c in df.columns if c not in chain_specific_cols]
        rows = []
        for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Exploding multichain entries",
            leave=False,
        ):
            row_dict = dict(zip(df.columns, row))
            for i in range(1, chain_n + 1):
                has_chain = False
                chain_data = {}
                for field in chain_fields:
                    col = f"{field} {i}"
                    val = row_dict.get(col)
                    chain_data[field] = val
                    if pd.notnull(val):
                        has_chain = True
                if has_chain:
                    new_row = {col: row_dict[col] for col in meta_cols}
                    for field in chain_fields:
                        new_row[f"{field} 1"] = chain_data[field]
                    rows.append(new_row)
        return pd.DataFrame(rows)

    def _to_float(self, x):
        try:
            return float(x)
        except:
            try:
                return float(x.strip().replace(",", "")[1:])
            except:
                self.logger.error(f"Failed to convert '{x}' to float.")
                return None

    def _clean_binding_affinity_df(self, binding_df):
        types = ["ki", "kd", "ec50", "ic50"]
        old_sizes = {
            _type: len(binding_df[binding_df["value_type"] == _type]) for _type in types
        }
        binding_df = binding_df[binding_df["value"] > 0].copy().reset_index(drop=True)
        binding_df["log10_value"] = np.log10(binding_df["value"])

        group_cols = [
            "smiles",
            "smiles_hash",
            "organism",
            "temp",
            "pH",
            "sequence",
            "uniprot_id",
            "value_type",
        ]
        first_cols = [
            "monomer_id",
            "ligand",
            "ligand_id",
            "target",
            "doi",
            "pmid",
            "ligand_target_complex_pdbs",
            "pubchem_cid",
            "pubchem_sid",
            "chebi_id",
            "chembl_id",
            "kegg_id",
            "zinc_id",
            "pdbs",
            "unit",
        ]

        grouped = binding_df.groupby(group_cols, dropna=False)
        first_values = grouped[first_cols].first()
        mean_values = grouped["log10_value"].mean()
        binding_df = first_values.join(mean_values).reset_index()

        binding_df["value"] = 10 ** binding_df["log10_value"]
        for _type in types:
            if old_sizes[_type] > 0:
                new_size = len(binding_df[binding_df["value_type"] == _type])
                self.logger.info(
                    f"Cleaned {old_sizes[_type] - new_size} entries from { _type} data "
                    f"({(old_sizes[_type] - new_size) / old_sizes[_type] * 100:.2f}%). "
                    f"New size: {new_size}"
                )
        return binding_df

    def _save_value_type_dataset_variants(self, binding_affinity_df):
        def _save_dataset(df, path, value_type, dataset_kind):
            self.logger.info(
                f"Saving {value_type} {dataset_kind} binding affinity dataset to {path}"
            )
            df.to_parquet(path, compression="brotli", index=False)
            if value_type != "unified":
                del df
                gc.collect()

        def _save_dataset_variants(df, dataset_kind, path_key):
            self.logger.info(
                f"Saving {dataset_kind} binding affinity datasets sequentially..."
            )
            for value_type in self.VALUE_TYPES:
                df_slice = (
                    df
                    if value_type == "unified"
                    else df.loc[df["value_type"] == value_type]
                )
                _save_dataset(
                    df_slice,
                    self.VALUE_TYPE_DATASET_PATHS[value_type][path_key],
                    value_type,
                    dataset_kind,
                )

        if self.SAVE_1D:
            _save_dataset_variants(binding_affinity_df, "1D", "1d")

        annotated_binding_affinity_df = None
        if self.SAVE_3D:
            annotated_binding_affinity_df = assign_experimental_and_af_pdbs(
                binding_affinity_df.copy()
            )
            _save_dataset_variants(annotated_binding_affinity_df, "3D", "3d")

        return annotated_binding_affinity_df

    def setup(self):
        bindingdb_df = self._read_bindingdb_data()
        self.logger.info("Extracting single-chain protein entries...")
        single_chain = (
            bindingdb_df[
                bindingdb_df[
                    "Number of Protein Chains in Target (>1 implies a multichain complex)"
                ]
                == 1
            ]
            .reset_index(drop=True)
            .drop(
                columns=[
                    item
                    for sublist in [
                        [
                            f"UniProt (SwissProt) Primary ID of Target Chain {i}",
                            f"PDB ID(s) of Target Chain {i}",
                        ]
                        for i in range(2, 51)
                    ]
                    for item in sublist
                ]
            )
        )
        self.logger.info(f"Single-chain entries shape: {single_chain.shape}")

        self.logger.info("Extracting multi-chain protein entries...")
        multi_chain = bindingdb_df[
            bindingdb_df[
                "Number of Protein Chains in Target (>1 implies a multichain complex)"
            ]
            > 1
        ].reset_index(drop=True)
        self.logger.info(f"Multi-chain entries shape: {multi_chain.shape}")

        self.logger.info(
            "Flattening multi-chain protein entries to single-chain format: creating one row per chain..."
        )
        multi_chain = self._explode_multichain_with_metadata(multi_chain)
        self.logger.info(
            f"Flattened multi-chain entries to single-chain shape: {multi_chain.shape}"
        )

        self.logger.info(
            "Concatenating single-chain and flattened multi-chain entries..."
        )
        bindingdb_df = (
            pd.concat([single_chain, multi_chain], ignore_index=True)
            .drop(
                columns=[
                    "Number of Protein Chains in Target (>1 implies a multichain complex)",
                    "kon (M-1-s-1)",
                    "koff (s-1)",
                ]
            )
            .dropna(
                subset=[
                    "Ligand SMILES",
                    "UniProt (SwissProt) Primary ID of Target Chain 1",
                ],
                how="any",
            )
            .reset_index(drop=True)
        )
        self.logger.info(f"Total entries after concatenation: {bindingdb_df.shape}")
        del single_chain, multi_chain

        self.logger.info(
            "Cleaning temperature data (Converting to Kelvin and imputing missing values with room temperature 298.15 K)..."
        )
        bindingdb_df["Temp (C)"] = bindingdb_df["Temp (C)"].apply(
            lambda x: float(x.split(" ")[0] if pd.notna(x) else x) + 273.15
        )
        bindingdb_df["Temp (C)"] = bindingdb_df["Temp (C)"].fillna(298.15)

        self.logger.info(
            "Cleaning pH data (Imputing missing values with room pH 7.0)..."
        )
        bindingdb_df["pH"] = bindingdb_df["pH"].fillna(7.0)

        for col in ("PMID", "PubChem CID", "PubChem SID"):
            self.logger.info(f"Cleaning {col} data...")
            bindingdb_df[col] = bindingdb_df[col].apply(
                lambda x: str(int(x)) if pd.notna(x) else x
            )

        self.logger.info("Cleaning BindingDB Ligand Name data...")
        bindingdb_df["BindingDB Ligand Name"] = bindingdb_df[
            "BindingDB Ligand Name"
        ].apply(lambda x: x.split("::")[0])

        for col in ("Ki (nM)", "Kd (nM)", "EC50 (nM)", "IC50 (nM)"):
            self.logger.info(f"Cleaning {col} data...")
            bindingdb_df[col] = bindingdb_df[col].apply(self._to_float)

        self.logger.info("Renaming columns...")
        bindingdb_df.rename(
            columns={
                "Ligand SMILES": "smiles",
                "BindingDB MonomerID": "monomer_id",
                "BindingDB Ligand Name": "ligand",
                "Target Name": "target",
                "Target Source Organism According to Curator or DataSource": "organism",
                "Ki (nM)": "ki",
                "IC50 (nM)": "ic50",
                "Kd (nM)": "kd",
                "EC50 (nM)": "ec50",
                "Temp (C)": "temp",
                "BindingDB Entry DOI": "doi",
                "PMID": "pmid",
                "PDB ID(s) for Ligand-Target Complex": "ligand_target_complex_pdbs",
                "PubChem CID": "pubchem_cid",
                "PubChem SID": "pubchem_sid",
                "ChEBI ID of Ligand": "chebi_id",
                "ChEMBL ID of Ligand": "chembl_id",
                "KEGG ID of Ligand": "kegg_id",
                "ZINC ID of Ligand": "zinc_id",
                "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
                "PDB ID(s) of Target Chain 1": "pdbs",
            },
            inplace=True,
        )

        mask = bindingdb_df["smiles"].str.contains(r"\|")
        bindingdb_df.loc[mask, "smiles"] = (
            bindingdb_df.loc[mask, "smiles"].str.split("|").str[0].str.strip()
        )

        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )
        unique_smiles = bindingdb_df[["smiles"]].drop_duplicates()
        self.logger.info("Canonicalizing unique SMILES...")
        unique_smiles["canonical_smiles"] = unique_smiles["smiles"].parallel_apply(
            canonicalize_smiles
        )
        unique_smiles = unique_smiles.dropna(subset=["canonical_smiles"]).reset_index(
            drop=True
        )
        unique_smiles["smiles_hash"] = unique_smiles["canonical_smiles"].parallel_apply(
            smiles_hash
        )

        old_full_size = len(bindingdb_df)
        old_ki_size = len(bindingdb_df[bindingdb_df["ki"].notna()])
        old_kd_size = len(bindingdb_df[bindingdb_df["kd"].notna()])
        old_ec50_size = len(bindingdb_df[bindingdb_df["ec50"].notna()])
        old_ic50_size = len(bindingdb_df[bindingdb_df["ic50"].notna()])
        bindingdb_df = (
            bindingdb_df.merge(
                unique_smiles,
                on="smiles",
                how="inner",
            )
            .dropna(subset=["canonical_smiles"])
            .drop(columns=["smiles"])
            .rename(columns={"canonical_smiles": "smiles"})
        )
        self.logger.info(
            f"Dropped {old_full_size - len(bindingdb_df)} entries due to invalid SMILES "
            f"({(old_full_size - len(bindingdb_df)) / old_full_size * 100:.2f}%). "
            f"New size: {len(bindingdb_df)}"
        )
        self.logger.info(
            f"Dropped {old_ki_size - len(bindingdb_df[bindingdb_df['ki'].notna()])} Ki entries "
            f"({(old_ki_size - len(bindingdb_df[bindingdb_df['ki'].notna()])) / old_ki_size * 100:.2f}%). "
            f"New Ki size: {len(bindingdb_df[bindingdb_df['ki'].notna()])}"
        )
        self.logger.info(
            f"Dropped {old_kd_size - len(bindingdb_df[bindingdb_df['kd'].notna()])} Kd entries "
            f"({(old_kd_size - len(bindingdb_df[bindingdb_df['kd'].notna()])) / old_kd_size * 100:.2f}%). "
            f"New Kd size: {len(bindingdb_df[bindingdb_df['kd'].notna()])}"
        )
        self.logger.info(
            f"Dropped {old_ec50_size - len(bindingdb_df[bindingdb_df['ec50'].notna()])} EC50 entries "
            f"({(old_ec50_size - len(bindingdb_df[bindingdb_df['ec50'].notna()])) / old_ec50_size * 100:.2f}%). "
            f"New EC50 size: {len(bindingdb_df[bindingdb_df['ec50'].notna()])}"
        )
        self.logger.info(
            f"Dropped {old_ic50_size - len(bindingdb_df[bindingdb_df['ic50'].notna()])} IC50 entries "
            f"({(old_ic50_size - len(bindingdb_df[bindingdb_df['ic50'].notna()])) / old_ic50_size * 100:.2f}%). "
            f"New IC50 size: {len(bindingdb_df[bindingdb_df['ic50'].notna()])}"
        )

        self.logger.info("Fetching protein sequences from UniProt...")
        bindingdb_df["uniprot_id"] = bindingdb_df["uniprot_id"].str.split()
        bindingdb_df = bindingdb_df.explode("uniprot_id").reset_index(drop=True)

        bindingdb_proteins = bindingdb_df["uniprot_id"].dropna().unique()

        bindingdb_df_uniprot = get_uniprot_acc_json(bindingdb_proteins)
        bindingdb_df_uniprot = pd.DataFrame(
            [i for i in bindingdb_df_uniprot.values() if i is not None]
        )[["acc_id", "sequence"]]

        old_size = len(bindingdb_df)
        bindingdb_df = bindingdb_df.merge(
            bindingdb_df_uniprot.rename(columns={"acc_id": "uniprot_id"}),
            on="uniprot_id",
            how="inner",
        )
        self.logger.info(
            f"Dropped {old_size - len(bindingdb_df)} entries due to missing protein sequences "
            f"({(old_size - len(bindingdb_df)) / old_size * 100:.2f}%). "
            f"New size: {len(bindingdb_df)}"
        )
        del bindingdb_df_uniprot, bindingdb_proteins, unique_smiles

        conditions = [
            (bindingdb_df["pubchem_cid"].notna()),
            (bindingdb_df["pubchem_cid"].isna() & bindingdb_df["chebi_id"].notna()),
            (
                bindingdb_df["pubchem_cid"].isna()
                & bindingdb_df["chebi_id"].isna()
                & bindingdb_df["kegg_id"].notna()
            ),
            (
                bindingdb_df["pubchem_cid"].isna()
                & bindingdb_df["chebi_id"].isna()
                & bindingdb_df["kegg_id"].isna()
                & bindingdb_df["chembl_id"].notna()
            ),
            (
                bindingdb_df["pubchem_cid"].isna()
                & bindingdb_df["chebi_id"].isna()
                & bindingdb_df["kegg_id"].isna()
                & bindingdb_df["chembl_id"].isna()
                & bindingdb_df["zinc_id"].notna()
            ),
        ]

        choices = [
            "PUBCHEM:C" + bindingdb_df["pubchem_cid"].astype(str),
            "CHEBI:" + bindingdb_df["chebi_id"].astype(str),
            "KEGG:" + bindingdb_df["kegg_id"].astype(str),
            "CHEMBL:" + bindingdb_df["chembl_id"].astype(str),
            "ZINC:" + bindingdb_df["zinc_id"].astype(str),
        ]

        bindingdb_df["ligand_id"] = np.select(
            conditions, choices, default=bindingdb_df["ligand"]
        )

        types = ["ki", "kd", "ec50", "ic50"]
        bindingdb_dfs = {}
        for _type in types:
            self.logger.info(f"Processing {_type} data...")
            bindingdb_dfs[_type] = (
                bindingdb_df[bindingdb_df[_type].notna()]
                .copy()
                .reset_index(drop=True)
                .drop(columns=[col for col in types if col != _type])
                .rename(columns={_type: "value"})
            )
            bindingdb_dfs[_type]["value_type"] = _type
            # Convert nM to mM (1 nM = 1e-6 mM)
            bindingdb_dfs[_type]["value"] = bindingdb_dfs[_type]["value"] * 1e-6
            bindingdb_dfs[_type]["unit"] = "millimolar (mM)"
            self.logger.info(f"{_type} dataset shape: {bindingdb_dfs[_type].shape}")

        binding_affinity_df = pd.concat(
            [bindingdb_dfs["kd"], bindingdb_dfs["ec50"], bindingdb_dfs["ic50"]],
            ignore_index=True,
        )
        self.logger.info(
            f"Unified affinities dataset shape: {binding_affinity_df.shape}"
        )
        binding_affinity_df = self._clean_binding_affinity_df(binding_affinity_df)
        self.logger.info(
            f"Cleaned and unified affinities dataset shape: {binding_affinity_df.shape}"
        )
        binding_affinity_df["value_type_embedding"] = (
            pd.get_dummies(binding_affinity_df["value_type"])
            .astype(int)
            .values.tolist()
        )

        binding_affinity_df["pdbs"] = (
            binding_affinity_df["pdbs"]
            .str.split(",")
            .apply(lambda x: [] if type(x) != list and pd.isna(x) else x)
        )
        binding_affinity_df = add_uniprot_date_column(
            binding_affinity_df, verbose=True
        )

        self.logger.info("Saving datasets to parquet files...")

        for _type in types:
            bindingdb_dfs[_type].to_parquet(
                self.BINDINGDB_DATA_PATH / f"{_type}.parquet",
                compression="brotli",
                index=False,
            )
            self.logger.info(
                f"Saved cleaned { _type} dataset to {self.BINDINGDB_DATA_PATH / f'{_type}.parquet'}"
            )

        annotated_binding_affinity_df = self._save_value_type_dataset_variants(
            binding_affinity_df
        )

        self.logger.info(
            f"Saving enzyme sequences to {self.BINDING_AFFINITY_SEQUENCES_PICKLE_PATH}"
        )
        with open(self.BINDING_AFFINITY_SEQUENCES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                binding_affinity_df[["uniprot_id", "sequence"]]
                .rename(columns={"uniprot_id": "acc_id"})
                .to_dict(orient="records"),
                f,
            )

        self.logger.info(
            f"Saving ligand SMILES to {self.BINDING_AFFINITY_SMILES_PICKLE_PATH}"
        )
        with open(self.BINDING_AFFINITY_SMILES_PICKLE_PATH, "wb") as f:
            pickle.dump(binding_affinity_df["smiles"].unique().tolist(), f)

        self.logger.info("All datasets saved successfully.")


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = BindingAffinityDatasetBuilder(cfg)
    builder.setup()
