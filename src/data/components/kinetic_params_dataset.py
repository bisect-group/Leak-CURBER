import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pickle
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize, compose

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
smiles_hash = chem_utils.smiles_hash
mutate_sequence = chem_utils.mutate_sequence
canonicalize_smiles = chem_utils.canonicalize_smiles
get_uniprot_acc_json = chem_utils.get_uniprot_acc_json
add_uniprot_date_column = chem_utils.add_uniprot_date_column
get_pubchem_compound = chem_utils.get_pubchem_compound
assign_experimental_and_af_pdbs = chem_utils.assign_experimental_and_af_pdbs
get_pubchem_cids_from_name_parallel = chem_utils.get_pubchem_cids_from_name_parallel


class KineticParamsDatasetBuilder:
    SUPPORTED_VALUE_TYPES = {"unified", "kcat", "km", "ki"}

    def __init__(self, cfg: DictConfig):
        self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.sabio_kinetic_params_parquet_file_path
        )
        self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.brenda_kinetic_params_parquet_file_path
        )
        self.METACYC_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.metacyc_kinetic_params_parquet_file_path
        )
        self.BINDINGDB_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.bindingdb_kinetic_params_parquet_file_path
        )

        self.UNIFIED_KINETIC_PARAMS_SMILES_PICKLE_PATH = Path(
            cfg.kinetic_params_dataset.unified_kinetic_params_smiles_pickle_path
        )
        self.UNIFIED_KINETIC_PARAMS_SEQUENCES_PICKLE_PATH = Path(
            cfg.kinetic_params_dataset.unified_kinetic_params_sequences_pickle_path
        )
        self.VALUE_TYPES = list(dict.fromkeys(cfg.kinetic_params_dataset.value_types))
        unsupported_value_types = set(self.VALUE_TYPES) - self.SUPPORTED_VALUE_TYPES
        if unsupported_value_types:
            raise ValueError(
                "Unsupported kinetic_params_dataset.value_types: "
                f"{sorted(unsupported_value_types)}. "
                f"Supported values: {sorted(self.SUPPORTED_VALUE_TYPES)}"
            )
        if not self.VALUE_TYPES:
            raise ValueError("kinetic_params_dataset.value_types must not be empty.")
        self.SAVE_1D = cfg.kinetic_params_dataset.save_1d_dataset
        self.SAVE_3D = cfg.kinetic_params_dataset.save_3d_dataset
        if not (self.SAVE_1D or self.SAVE_3D):
            raise ValueError(
                "At least one of kinetic_params_dataset.save_1d_dataset or "
                "kinetic_params_dataset.save_3d_dataset must be true."
            )

        self.UNIFIED_KINETIC_PARAMS_1D_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.unified_kinetic_params_1d_parquet_file_path
        )
        self.UNIFIED_KINETIC_PARAMS_3D_PARQUET_FILE_PATH = Path(
            cfg.kinetic_params_dataset.unified_kinetic_params_3d_parquet_file_path
        )
        self.VALUE_TYPE_DATASET_PATHS = {
            value_type: {
                "1d": Path(
                    cfg.kinetic_params_dataset.dataset_paths[value_type].parquet_1d
                ),
                "3d": Path(
                    cfg.kinetic_params_dataset.dataset_paths[value_type].parquet_3d
                ),
            }
            for value_type in self.VALUE_TYPES
        }

        LOG_PATH = Path(cfg.kinetic_params_dataset.log_dir)

        for path in [
            self.UNIFIED_KINETIC_PARAMS_SMILES_PICKLE_PATH.parent,
            self.UNIFIED_KINETIC_PARAMS_SEQUENCES_PICKLE_PATH.parent,
            LOG_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        for dataset_paths in self.VALUE_TYPE_DATASET_PATHS.values():
            for path in dataset_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.kinetic_params_dataset.log_file_name
        ).get_logger()

        for filepath in [
            self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH,
            self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH,
            self.METACYC_KINETIC_PARAMS_PARQUET_FILE_PATH,
            self.BINDINGDB_KINETIC_PARAMS_PARQUET_FILE_PATH,
        ]:
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

    def _read_kinetic_law_dfs(self):
        """
        Read the kinetic law DataFrames parsed by the respective parsing scripts.
        """
        self.logger.info("Reading SABIO kinetic law DataFrames...")
        try:
            sabio_kinetic_params = pd.read_parquet(
                self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH
            )
        except Exception as e:
            self.logger.error(f"Error while reading SABIO kinetic law DataFrames: {e}")
            raise e
        self.logger.info("SABIO kinetic law DataFrames read successfully.")

        self.logger.info("Adding source DB column to SABIO DataFrames...")
        sabio_kinetic_params["source"] = "sabio"

        self.logger.info("Reading BRENDA kinetic law DataFrames...")
        try:
            brenda_kinetic_params = pd.read_parquet(
                self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH
            )
        except Exception as e:
            self.logger.error(f"Error while reading BRENDA kinetic law DataFrames: {e}")
            raise e
        self.logger.info("BRENDA kinetic law DataFrames read successfully.")

        self.logger.info("Adding source DB column to BRENDA DataFrames...")
        brenda_kinetic_params["source"] = "brenda"

        return [brenda_kinetic_params, sabio_kinetic_params]

    def _clean_kinetic_law_df(self, kinetic_law_df, kinetic_law):
        self.logger.info(
            f"No. of rows before cleaning {kinetic_law} DataFrame: {len(kinetic_law_df)}"
        )
        self.logger.info(f"Removing rows with non-positive values...")
        kinetic_law_df["value"] = kinetic_law_df["value"].astype(float)
        kinetic_law_df = kinetic_law_df[kinetic_law_df["value"] > 0].copy()
        self.logger.info(
            f"Shape after removing non-positive values: {kinetic_law_df.shape}"
        )

        self.logger.info(f"Adding log10 transformation of {kinetic_law} values...")
        kinetic_law_df[f"log10_value"] = np.log10(kinetic_law_df["value"])
        self.logger.info(
            f"Log10 transformation of {kinetic_law} values added successfully."
        )

        # Convert Temperature from °C to K where applicable and then fill missing Temperature values with 298.15 K (25°C)
        mask = kinetic_law_df["Temperature"].notna()
        kinetic_law_df.loc[mask, "Temperature"] = (
            kinetic_law_df.loc[mask, "Temperature"].astype(float) + 273.15
        )
        kinetic_law_df["Temperature"] = kinetic_law_df["Temperature"].fillna(298.15)

        # Fill missing pH values with 7.0
        kinetic_law_df["pH"] = kinetic_law_df["pH"].fillna(7.0)

        try:
            group_cols = [
                "smiles",
                "enzymeType",
                "Organism",
                "uniprot_id",
                "sequence",
                "Temperature",
                "pH",
                "unit",
            ]
            if kinetic_law == "kcat":
                kinetic_law_df = kinetic_law_df.loc[
                    kinetic_law_df.groupby(group_cols, dropna=False)[
                        f"log10_value"
                    ].idxmax()
                ].reset_index(drop=True)
            else:
                kinetic_law_df = (
                    kinetic_law_df.groupby(group_cols, dropna=False)
                    .agg(
                        {
                            f"log10_value": "mean",
                            "ECNumber": "first",
                            "substrate": "first",
                            "substrate_id": "first",
                            "source": lambda x: "|".join(set(x)),
                            "mutations": "first",
                            "pdbs": lambda x: next(
                                (pdb for pdb in x if pdb is not None and pdb != []),
                                None,
                            ),
                        }
                    )
                    .reset_index()
                )
                kinetic_law_df["value"] = (10 ** kinetic_law_df["log10_value"]).round(2)
        except Exception as e:
            self.logger.error(f"Error while cleaning {kinetic_law} DataFrame: {e}")
            raise e
        self.logger.info(
            f"Shape after cleaning {kinetic_law} DataFrame: {kinetic_law_df.shape}"
        )

        kinetic_law_df["value_type"] = kinetic_law
        return kinetic_law_df

    def _unify_brenda_sabio(self):
        """
        Unify the kinetic law DataFrames from BRENDA and SABIO.
        """
        unified_df = pd.concat(self._read_kinetic_law_dfs(), ignore_index=True)
        self.logger.info(
            "Searching PubChem for CID for compounds that do not have SMILES yet..."
        )
        try:
            # Only get substrates where smiles is NaN
            substrates = (
                unified_df.loc[unified_df["smiles"].isna(), "substrate"]
                .unique()
                .tolist()
            )
            compound_cid_df = pd.DataFrame(
                get_pubchem_cids_from_name_parallel(substrates)
            )
            compound_to_cid = dict(
                compound_cid_df.dropna(subset=["cid"])[["compound_name", "cid"]]
                .drop_duplicates()
                .values
            )
            cids = list(set(compound_to_cid.values()))
            self.logger.info("PubChem CIDs retrieved successfully.")
        except Exception as e:
            self.logger.error(f"Error while retrieving PubChem CIDs: {e}")
            raise e

        self.logger.info("Retrieving SMILES information from PubChem...")
        try:
            compounds = {
                cid: cpd["smiles"] for cid, cpd in get_pubchem_compound(cids).items()
            }
        except Exception as e:
            self.logger.error(
                f"Error while retrieving SMILES information from PubChem: {e}"
            )
            raise e
        self.logger.info("SMILES information retrieved successfully from PubChem.")

        self.logger.info("Downloading sequences from UniProt...")
        try:
            uniprot_ids = unified_df["UniprotID"].dropna().unique().tolist()
            sequences = get_uniprot_acc_json(uniprot_ids, verbose=True)
            sequences = pd.DataFrame(
                [elem for elem in sequences.values() if elem]
            ).rename(columns={"acc_id": "UniprotID"})
            self.logger.info("Sequences downloaded successfully.")
        except Exception as e:
            self.logger.error(f"Error while downloading sequences from UniProt: {e}")
            raise e

        self.logger.info("Adding missing SMILES to kinetic law DataFrames...")
        unified_df.loc[unified_df["smiles"].isna(), "smiles"] = (
            unified_df.loc[unified_df["smiles"].isna(), "substrate"]
            .map(compound_to_cid)
            .map(compounds)
        )
        self.logger.info("Missing SMILES added successfully.")

        self.logger.info("Merging sequences with kinetic law DataFrames...")
        unified_df = unified_df.merge(sequences, on="UniprotID", how="left")
        self.logger.info("Sequences merged successfully.")

        unified_df["mutations"] = unified_df["mutations"].apply(
            lambda x: tuple(x) if x is not None else x
        )
        self.logger.info("Mutating sequences where mutations are present...")
        unified_df.loc[unified_df["mutations"].notna()] = unified_df.loc[
            unified_df["mutations"].notna()
        ].apply(mutate_sequence, axis=1)
        self.logger.info("Sequences mutated successfully.")

        self.logger.info(
            "BRENDA and SABIO kinetic law DataFrames unified successfully."
        )
        unified_df["value_type"] = unified_df["value_type"].str.lower()

        return (
            unified_df.drop(columns=["ecs"])
            .dropna(subset=["sequence", "smiles"])
            .drop_duplicates(subset=["smiles", "sequence", "value", "value_type"])
            .reset_index(drop=True)
        )

    def _unify_metacyc(self, sabio_brenda_df):
        def prepare_metacyc_df(df):
            df["Temperature"] = None
            df["pH"] = None
            df["mutations"] = None
            df["pdbs"] = None
            df["enzymeType"] = "wildtype"
            df["source"] = "metacyc"
            df = df[
                [
                    "Organism",
                    "UniprotID",
                    "ECNumber",
                    "value",
                    "unit",
                    "Temperature",
                    "substrate",
                    "substrate_id",
                    "enzymeType",
                    "source",
                    "pH",
                    "mutations",
                    "smiles",
                    "pdbs",
                    "sequence",
                    "value_type",
                ]
            ]
            return df

        metacyc_kinetic_params = prepare_metacyc_df(
            df=pd.read_parquet(self.METACYC_KINETIC_PARAMS_PARQUET_FILE_PATH)
        )
        self.logger.info(
            f"Metacyc kinetic params shape: {metacyc_kinetic_params.shape}"
        )

        unified_df = (
            pd.concat([sabio_brenda_df, metacyc_kinetic_params], ignore_index=True)
            .dropna(subset=["smiles", "sequence"])
            .drop_duplicates(subset=["smiles", "sequence", "value", "value_type"])
            .reset_index(drop=True)
        )

        unified_df["UniprotID"] = unified_df["UniprotID"].str.replace(
            "/", "_", regex=False
        )

        return unified_df.rename(columns={"UniprotID": "uniprot_id"})

    def _unify_bindingdb(self, unified_df):
        def prepare_bindingdb_df(df):
            df = df[
                [
                    "ligand_id",
                    "ligand",
                    "value",
                    "organism",
                    "uniprot_id",
                    "value_type",
                    "unit",
                    "pH",
                    "temp",
                    "smiles",
                    "pdbs",
                    "sequence",
                ]
            ].rename(
                columns={
                    "organism": "Organism",
                    "temp": "Temperature",
                    "ligand": "substrate",
                    "ligand_id": "substrate_id",
                }
            )
            df["value_type"] = df["value_type"].str.lower()
            df["enzymeType"], df["source"] = "wildtype", "bindingdb"
            df["ECNumber"], df["mutations"] = None, None
            df["pdbs"] = df["pdbs"].apply(lambda x: x.split(",") if pd.notna(x) else x)
            return df

        bindingdb_ki = prepare_bindingdb_df(
            df=pd.read_parquet(self.BINDINGDB_KINETIC_PARAMS_PARQUET_FILE_PATH)
        )
        unified_df = (
            pd.concat([unified_df, bindingdb_ki], ignore_index=True)
            .dropna(subset=["smiles", "sequence"])
            .drop_duplicates(subset=["smiles", "sequence", "value", "value_type"])
            .reset_index(drop=True)
        )
        return unified_df

    def _clean_unified_df(self, unified_df):
        import os
        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=False)

        old_sizes = {"full": len(unified_df)}
        for v_type in unified_df["value_type"].unique():
            old_sizes[v_type] = (unified_df["value_type"] == v_type).sum()

        self.logger.info("Validating SMILES strings in unified DataFrame...")
        unique_smiles_df = unified_df[["smiles"]].drop_duplicates().copy()
        unique_smiles_df["canonical_smiles"] = unique_smiles_df[
            "smiles"
        ].parallel_apply(canonicalize_smiles)
        unified_df = (
            unified_df.merge(unique_smiles_df, on="smiles", how="inner")
            .dropna(subset=["canonical_smiles"])
            .drop(columns=["smiles"])
            .rename(columns={"canonical_smiles": "smiles"})
        )

        for v_type in unified_df["value_type"].unique():
            self.logger.info(
                f"Dropped {old_sizes[v_type] - (unified_df['value_type'] == v_type).sum()} {v_type} entries due to invalid SMILES "
                f"({(old_sizes[v_type] - (unified_df['value_type'] == v_type).sum()) / old_sizes[v_type] * 100:.2f}%). "
                f"New size: {(unified_df['value_type'] == v_type).sum()}"
            )

        unified_df = pd.concat(
            [
                self._clean_kinetic_law_df(
                    unified_df[unified_df["value_type"] == "kcat"].copy(), "kcat"
                ),
                self._clean_kinetic_law_df(
                    unified_df[unified_df["value_type"] == "km"].copy(), "km"
                ),
                self._clean_kinetic_law_df(
                    unified_df[unified_df["value_type"] == "ki"].copy(), "ki"
                ),
            ],
            ignore_index=True,
        )

        self.logger.info("Adding value type embeddings...")
        unified_df["value_type_embedding"] = (
            pd.get_dummies(unified_df["value_type"]).astype(int).values.tolist()
        )
        unified_df["uniprot_id"] = unified_df["uniprot_id"].apply(
            lambda x: x.replace(" ", "_")
        )

        self.logger.info("Adding SMILES hashes...")
        unique_smiles = unified_df[["smiles"]].drop_duplicates()
        unique_smiles["smiles_hash"] = unique_smiles["smiles"].parallel_apply(
            smiles_hash
        )
        unified_df = unified_df.merge(unique_smiles, on="smiles", how="left")
        return unified_df

    def _pickle_all_sequences(self, unified_df):
        """
        Serialize and save all unique protein sequences from the provided kinetic law DataFrames
        for use in downstream embedding pipelines. Each sequence is associated with a unique
        accession identifier (acc_id), which is either the UniProt ID for wildtype enzymes or
        a combination of UniProt ID and mutation(s) for mutant enzymes.

        Args:
            unified_df (pd.DataFrame): DataFrame containing unified kinetic law data.

        Output:
            Writes a pickle file (kinetic_sequences.pkl) to DATA_PATH containing a list of
            dictionaries, each with 'acc_id' and 'sequence' keys.
        """
        sequences_dict = (
            unified_df[["uniprot_id", "sequence"]]
            .dropna(subset=["sequence"])
            .rename(columns={"uniprot_id": "acc_id"})
            .drop_duplicates()
            .to_dict(orient="records")
        )

        with open(self.UNIFIED_KINETIC_PARAMS_SEQUENCES_PICKLE_PATH, "wb") as f:
            pickle.dump(sequences_dict, f)

        return sequences_dict

    def _pickle_all_smiles(self, unified_df):
        """
        Serialize and save all unique SMILES strings from the provided kinetic law DataFrames
        for use in downstream embedding pipelines. Each SMILES string is associated with a
        unique substrate name.

        Args:
            unified_df (pd.DataFrame): DataFrame containing unified kinetic law data.

        Output:
            Writes a pickle file (kinetic_smiles.pkl) to DATA_PATH containing a list of
            dictionaries, each with 'substrate' and 'smiles' keys.
        """
        with open(self.UNIFIED_KINETIC_PARAMS_SMILES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                unified_df[["substrate", "smiles"]]
                .dropna(subset=["smiles"])
                .drop_duplicates()
                .to_dict(orient="records"),
                f,
            )

    def _save_value_type_dataset_variants(self, unified_df):
        def _iter_value_type_dfs(df):
            for value_type in self.VALUE_TYPES:
                if value_type == "unified":
                    yield value_type, df
                    continue
                yield value_type, df.loc[df["value_type"] == value_type]

        def _save_dataset(df, path, value_type, dataset_kind):
            self.logger.info(
                f"Saving {value_type} {dataset_kind} kinetic params dataset to {path}..."
            )
            df.to_parquet(path, index=False, compression="brotli")
            if value_type != "unified":
                del df
                gc.collect()

        if self.SAVE_1D:
            for value_type, df_slice in _iter_value_type_dfs(unified_df):
                _save_dataset(
                    df_slice,
                    self.VALUE_TYPE_DATASET_PATHS[value_type]["1d"],
                    value_type,
                    "1D",
                )

        annotated_unified_df = None
        if self.SAVE_3D:
            annotated_unified_df = assign_experimental_and_af_pdbs(unified_df.copy())
            for value_type, df_slice in _iter_value_type_dfs(annotated_unified_df):
                _save_dataset(
                    df_slice,
                    self.VALUE_TYPE_DATASET_PATHS[value_type]["3d"],
                    value_type,
                    "3D",
                )

        return annotated_unified_df

    def setup(self):

        self.logger.info(
            "Starting unification of BRENDA and SABIO kinetic law DataFrames..."
        )
        unified_df = self._unify_brenda_sabio()
        for value_type in ["kcat", "km", "ki"]:
            self.logger.info(
                f"No. of {value_type} entries after BRENDA/SABIO unification: {(unified_df['value_type'] == value_type).sum()}"
            )

        self.logger.info("Unifying with Metacyc kinetic law DataFrames...")
        unified_df = self._unify_metacyc(unified_df)
        self.logger.info(
            "Unification with Metacyc kinetic law DataFrames completed successfully"
        )
        for value_type in ["kcat", "km"]:
            self.logger.info(
                f"No. of {value_type} entries after Metacyc unification: {(unified_df['value_type'] == value_type).sum()}"
            )

        self.logger.info("Unifying with BindingDB kinetic law DataFrames...")
        unified_df = self._unify_bindingdb(unified_df)
        self.logger.info(
            "Unification with BindingDB kinetic law DataFrames completed successfully"
        )
        self.logger.info(
            f"No. of ki entries after BindingDB unification: {(unified_df['value_type'] == 'ki').sum()}"
        )

        self.logger.info("Cleaning unified kinetic law DataFrames...")
        unified_df = self._clean_unified_df(unified_df)

        self.logger.info("Pickling SMILES for downstream embedding pipelines...")
        self._pickle_all_smiles(unified_df)
        self.logger.info("Pickling of SMILES completed successfully")

        self.logger.info("Pickling sequences for downstream embedding pipelines...")
        sequences_dict = self._pickle_all_sequences(unified_df)
        self.logger.info("Pickling of sequences completed successfully")
        unified_df = add_uniprot_date_column(unified_df, verbose=True)

        try:
            annotated_unified_df = self._save_value_type_dataset_variants(unified_df)
        except Exception as e:
            self.logger.error(
                f"Error while saving unified kinetic law DataFrame(s): {e}"
            )
            raise e
        self.logger.info(
            "Unification of BRENDA, SABIO, Metacyc kinetic law DataFrames completed successfully."
        )

        return annotated_unified_df if self.SAVE_3D else unified_df


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs/"):
        cfg = compose(config_name="data_processing")
    builder = KineticParamsDatasetBuilder(cfg)
    unified_kinetic_params_df = builder.setup()
