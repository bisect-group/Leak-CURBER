import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import io
import re
import requests
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from hydra import compose, initialize

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger
from src.data.components.chebi import ChEBIDatasetUtils

chem_utils = ChemUtils()
download = chem_utils.download
get_uniprot_acc_json = chem_utils.get_uniprot_acc_json
get_uniprot_ec_acc_map = chem_utils.get_uniprot_ec_acc_map
parse_all_reference_xmls = chem_utils.parse_all_reference_xmls
download_pubmed_abstracts_parallel = chem_utils.download_pubmed_abstracts_parallel
normalize_ec_collection = chem_utils.normalize_ec_collection

chebi_utils = ChEBIDatasetUtils()
parse_chebi_jsons = chebi_utils.parse_chebi_jsons
download_chebi_json = chebi_utils.download_chebi_json


class RheaDatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.RHEA_URL = cfg.rhea.rhea_url
        self.RHEA_REACTIONS_URL = cfg.rhea.rhea_reactions_url
        self.RHEA_UNIPROT_URL = cfg.rhea.rhea_uniprot_url

        self.RHEA_REACTIONS_FILE_PATH = Path(cfg.rhea.rhea_reactions_file_path)
        self.RHEA_UNIPROT_FILE_PATH = Path(cfg.rhea.rhea_uniprot_file_path)

        self.RHEA_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.rhea.rhea_reactions_parquet_file_path
        )
        self.RHEA_REFERENCES_PARQUET_FILE_PATH = Path(
            cfg.rhea.rhea_references_parquet_file_path
        )

        LOG_PATH = Path(cfg.rhea.log_dir)

        for path in [
            LOG_PATH,
            self.RHEA_REACTIONS_FILE_PATH.parent,
            self.RHEA_UNIPROT_FILE_PATH.parent,
            self.RHEA_REACTIONS_PARQUET_FILE_PATH.parent,
            self.RHEA_REFERENCES_PARQUET_FILE_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.rhea.log_file_name
        ).get_logger()

    def extract_chebi_ids_from_rhea(self, rhea_entries, verbose=False):
        """
        Extract all unique CHEBI:xxxxxxx IDs from a Rhea reactions.txt file.

        Args:
            rhea_entries (list): List of Rhea entries as strings.
            verbose (bool): If True, print additional information.

        Returns:
            set: Set of unique CHEBI IDs as strings (e.g., 'CHEBI:16459').
        """
        self.logger.info("Extracting CHEBI IDs from Rhea entries...")
        chebi_pattern = re.compile(r"CHEBI:\d+")
        chebi_ids = set()
        for line in tqdm(
            rhea_entries, desc="Extracting CHEBI IDs", disable=not verbose, leave=False
        ):
            chebi_ids.update(chebi_pattern.findall(line))
        self.logger.info(f"Found {len(chebi_ids)} unique CHEBI IDs.")
        self.logger.debug(f"Extracted CHEBI IDs: {list(chebi_ids)[:10]}...")
        return chebi_ids

    def parse_rhea_entry(self, entry, keep_chebi_ids=False):
        """Parse a single Rhea entry and extract relevant information.

        Args:
            entry (str): A single Rhea entry as a string.
            keep_chebi_ids (bool): If True, include CHEBI IDs in the output.

        Returns:
            dict: A dictionary containing parsed information from the Rhea entry.
        """

        def split_coefficients(metabolite, return_coefficient=True):
            """
            Split a metabolite string into its coefficient and name.

            Args:
                metabolite (str): The metabolite string, e.g., "2 n-glucose".
                return_coefficient (bool): If True, return both coefficient and name.

            Returns:
                tuple: A tuple containing the coefficient and name (if return_coefficient is True), or just the name.
            """
            pattern = re.compile(r"^\s*(?:(\d+n|\(n[+-]\d+\)|\d+|n)\s+)?(.*)$")
            match = pattern.match(metabolite)
            coefficient = match.group(1) if match.group(1) else "1"
            metabolite_name = match.group(2)
            return (
                [coefficient, metabolite_name]
                if return_coefficient
                else metabolite_name
            )

        def get_formulae_smiles(metabolites, chebi_ids, chebi_jsons):
            """
            Get the formulae and SMILES for a list of metabolites based on their CHEBI IDs.

            Args:
                metabolites (list): List containing metabolite names.
                chebi_ids (dict): Dictionary mapping metabolite names to CHEBI IDs.
                chebi_jsons (dict): Dictionary containing parsed CHEBI JSON data.

            Returns:
                tuple: A tuple containing two lists: formulae and SMILES.
            """
            formulae, smiles = [], []
            for m in metabolites:
                try:
                    metabolite = chebi_jsons[chebi_ids[m[1]]]
                except KeyError:
                    self.logger.info(f"Chebi ID {chebi_ids[m[1]]} not found.")
                    return [None], [None]
                formula = metabolite["formulae"][0] if metabolite["formulae"] else None
                formulae.append(f"{m[0]} {formula}" if formula else None)
                smiles.append(metabolite["smiles"])
            return formulae, smiles

        entry = re.split(r"\n+", entry.strip())

        ec_raw = re.split(r"\s+", entry[3], 1)[1] if len(entry) > 3 else None
        ec = normalize_ec_collection(ec_raw, fallback=None)

        definition = re.split(r"\s+", entry[1], 1)[1].replace("hnu", "photon")
        chebi_ids = re.split(r"\s+", entry[2], 1)[1]

        # identify reactions with macromolecules and remove them
        if "," in chebi_ids:
            return

        if " => " in definition:
            direction = "LEFT-TO-RIGHT"
            definition = definition.replace(" => ", " = ")
            chebi_ids = chebi_ids.replace(" => ", " = ")
        elif " <=> " in definition:
            direction = "REVERSIBLE"
            definition = definition.replace(" <=> ", " = ")
            chebi_ids = chebi_ids.replace(" <=> ", " = ")
        else:
            direction = None

        sub_eqn, prod_eqn = definition.split(" = ")
        sub_eqn = [split_coefficients(m) for m in sub_eqn.split(" + ")]
        prod_eqn = [split_coefficients(m) for m in prod_eqn.split(" + ")]

        sub_chebi_ids, prod_chebi_ids = chebi_ids.split(" = ")
        sub_chebi_ids = [
            split_coefficients(m, return_coefficient=False)
            for m in sub_chebi_ids.split(" + ")
        ]
        prod_chebi_ids = [
            split_coefficients(m, return_coefficient=False)
            for m in prod_chebi_ids.split(" + ")
        ]

        sub_chebi_ids = {
            name[1]: chebi_id for name, chebi_id in zip(sub_eqn, sub_chebi_ids)
        }
        prod_chebi_ids = {
            name[1]: chebi_id for name, chebi_id in zip(prod_eqn, prod_chebi_ids)
        }

        chebi_jsons = parse_chebi_jsons(
            set(list(sub_chebi_ids.values()) + list(prod_chebi_ids.values())),
            return_as_dict=True,
        )

        sub_formulae, sub_smiles = get_formulae_smiles(
            sub_eqn, sub_chebi_ids, chebi_jsons
        )
        prod_formulae, prod_smiles = get_formulae_smiles(
            prod_eqn, prod_chebi_ids, chebi_jsons
        )

        equation = (
            f"{' + '.join(sub_formulae)} = {' + '.join(prod_formulae)}"
            if None not in sub_formulae and None not in prod_formulae
            else None
        )
        rxn_smiles = (
            f"{'.'.join(sub_smiles)}>>{'.'.join(prod_smiles)}"
            if None not in sub_smiles and None not in prod_smiles
            else None
        )

        rhea_entry = {
            "rhea_id": re.split(r"\s+", entry[0], 1)[1],
            "direction": direction,
            "definition": definition,
            "chebi_ids": chebi_ids,
            "equation": equation,
            "rxn_smiles": rxn_smiles,
            "ec": ec,
        }

        if not keep_chebi_ids:
            rhea_entry.pop("chebi_ids")

        return rhea_entry

    def download_rhea_xref(self, id):
        """
        Download Rhea cross-references for a given Rhea ID.

        Args:
            id (str): The Rhea ID to query.

        Returns:
            pd.DataFrame: A DataFrame containing the cross-references for the specified Rhea ID
        """
        columns = "rhea-id,go,pubmed,reaction-xref(EcoCyc),reaction-xref(MetaCyc),reaction-xref(KEGG),reaction-xref(Reactome),reaction-xref(M-CSA)"
        params = {"query": id, "columns": columns, "format": "tsv"}
        rhea_reactions = (
            pd.read_csv(io.StringIO(requests.get(self.RHEA_URL, params).text), sep="\t")
            .sort_values("Reaction identifier")
            .reset_index(drop=True)
        )
        rhea_reactions.rename(
            columns={
                "Reaction identifier": "rhea_id",
                "Gene Ontology": "go_id",
                "PubMed": "pubmed_id",
                "Cross-reference (EcoCyc)": "ecocyc_id",
                "Cross-reference (MetaCyc)": "metacyc_id",
                "Cross-reference (KEGG)": "kegg_id",
                "Cross-reference (Reactome)": "reactome_id",
                "Cross-reference (M-CSA)": "m_csa_id",
            },
            inplace=True,
        )

        rhea_reactions["pubmed_id"] = (
            rhea_reactions["pubmed_id"]
            .str.split(";")
            .apply(lambda x: tuple(x) if isinstance(x, list) else x)
        )
        rhea_reactions["go_id"] = rhea_reactions["go_id"].str[3:10]
        rhea_reactions["ecocyc_id"] = rhea_reactions["ecocyc_id"].str[7:]
        rhea_reactions["metacyc_id"] = rhea_reactions["metacyc_id"].str[8:]
        rhea_reactions["kegg_id"] = rhea_reactions["kegg_id"].str[5:]
        rhea_reactions["reactome_id"] = rhea_reactions["reactome_id"].str.replace(
            "Reactome:", ""
        )
        rhea_reactions["reactome_id"] = (
            rhea_reactions["reactome_id"]
            .str.split(",")
            .apply(lambda x: tuple(x) if isinstance(x, list) else x)
        )

        return rhea_reactions

    def setup(self):
        """
        Set up the Rhea dataset generation process.
        This function downloads the necessary Rhea data files, parses the Rhea entries,
        and saves the processed data to a parquet file.
        """

        try:
            self.logger.info("Downloading Rhea reactions...")
            download(
                url=self.RHEA_REACTIONS_URL,
                path=self.RHEA_REACTIONS_FILE_PATH,
                overwrite=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to download Rhea reactions: {e}")
            return

        try:
            self.logger.info("Downloading Rhea - SwissProt cross-references...")
            download(
                url=self.RHEA_UNIPROT_URL,
                path=self.RHEA_UNIPROT_FILE_PATH,
                overwrite=True,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to download Rhea-SwissProt cross-references: {e}"
            )
            return

        self.logger.info("Parsing Rhea entries...")
        try:
            with open(self.RHEA_REACTIONS_FILE_PATH, "r") as f:
                rhea_entries = [
                    entry.strip()
                    for entry in f.read().strip().split("///")
                    if entry.strip()
                ]
            self.logger.debug(f"Loaded {len(rhea_entries)} Rhea entries.")
        except Exception as e:
            self.logger.error(f"Failed to read Rhea reactions file: {e}")
            return

        try:
            download_chebi_json(
                self.extract_chebi_ids_from_rhea(rhea_entries), verbose=True
            )
        except Exception as e:
            self.logger.error(f"Failed to download CHEBI JSONs: {e}")
            return

        self.logger.info("Parsing individual Rhea entries...")
        try:
            rhea_reactions = [
                self.parse_rhea_entry(entry)
                for entry in tqdm(
                    rhea_entries, desc="Parsing Rhea entries", leave=False
                )
            ]
            self.logger.debug(
                f"Parsed {len([r for r in rhea_reactions if r])} valid Rhea entries."
            )
        except Exception as e:
            self.logger.error(f"Error during parsing Rhea entries: {e}")
            return

        try:
            rhea_reactions_df = pd.DataFrame(
                [reaction for reaction in rhea_reactions if reaction]
            )
            self.logger.info(
                f"Created DataFrame with {len(rhea_reactions_df)} reactions."
            )
            rhea_reactions_df = rhea_reactions_df.merge(
                self.download_rhea_xref(""), on="rhea_id", how="left"
            )
        except Exception as e:
            self.logger.error(f"Failed to merge Rhea cross-references: {e}")
            return

        try:
            with open(self.RHEA_UNIPROT_FILE_PATH, "r") as f:
                rhea_uniprot = (
                    pd.read_csv(f, sep="\t")
                    .drop(columns=["DIRECTION", "RHEA_ID"])
                    .rename(columns={"MASTER_ID": "rhea_id", "ID": "uniprot_id"})
                )
            self.logger.debug(f"Loaded {len(rhea_uniprot)} Rhea-UniProt mappings.")
        except Exception as e:
            self.logger.error(f"Failed to read Rhea-UniProt file: {e}")
            return

        try:
            rhea_uniprot["rhea_id"] = "RHEA:" + rhea_uniprot["rhea_id"].astype(str)
            rhea_uniprot = (
                rhea_uniprot.groupby("rhea_id").agg({"uniprot_id": list}).reset_index()
            )
            rhea_uniprot["uniprot_id"] = rhea_uniprot["uniprot_id"].apply(
                lambda x: list(set(x))
            )
            rhea_reactions_df = rhea_reactions_df.merge(
                rhea_uniprot, on="rhea_id", how="left"
            )
            self.logger.debug("Merged UniProt IDs into reactions DataFrame.")
        except Exception as e:
            self.logger.error(f"Failed to merge UniProt IDs: {e}")
            return

        try:
            for col in [
                "ec",
                "go_id",
                "pubmed_id",
                "ecocyc_id",
                "metacyc_id",
                "kegg_id",
                "reactome_id",
                "m_csa_id",
                "uniprot_id",
            ]:
                rhea_reactions_df[col] = rhea_reactions_df.groupby(
                    rhea_reactions_df.index // 4
                )[col].transform("first")

            rhea_reactions_df.loc[rhea_reactions_df.index % 4 != 3, "kegg_id"] = None
            rhea_reactions_df.loc[rhea_reactions_df.index % 4 == 3, "reactome_id"] = (
                None
            )
            rhea_reactions_df.loc[rhea_reactions_df.index % 4 == 0, "m_csa_id"] = None
            rhea_reactions_df.loc[rhea_reactions_df.index % 4 == 3, "m_csa_id"] = None
            rhea_reactions_df.loc[rhea_reactions_df.index % 4 == 2, "direction"] = (
                "RIGHT-TO-LEFT"
            )
            self.logger.debug("Cleaned up DataFrame columns.")
        except Exception as e:
            self.logger.error(f"Error during DataFrame column cleanup: {e}")
            return

        self.logger.info("Mapping EC numbers to UniProt accession IDs...")
        try:
            rhea_reactions_df = rhea_reactions_df.explode("ec")
            mask = (
                rhea_reactions_df["ec"].notna() & rhea_reactions_df["uniprot_id"].isna()
            )
            ec_but_no_uniprot_acc_ids = sorted(
                set(rhea_reactions_df[mask]["ec"].tolist())
            )
            self.logger.debug(
                f"ECs without UniProt IDs: {ec_but_no_uniprot_acc_ids[:10]}..."
            )
            ec_uniprot_data = get_uniprot_ec_acc_map(
                ec_but_no_uniprot_acc_ids, verbose=True
            )
            rhea_reactions_df.loc[mask, "uniprot_id"] = rhea_reactions_df.loc[
                mask, "ec"
            ].map(ec_uniprot_data)
        except Exception as e:
            self.logger.error(f"Error mapping EC to UniProt IDs: {e}")
            return

        self.logger.info("Mapping UniProt accession IDs to EC numbers...")
        rhea_reactions_df = rhea_reactions_df.explode("uniprot_id")

        mask = rhea_reactions_df["ec"].isna() & rhea_reactions_df["uniprot_id"].notna()
        no_ec_but_uniprot_acc_ids = sorted(
            set(rhea_reactions_df[mask]["uniprot_id"].tolist())
        )

        acc_uniprot_data = get_uniprot_acc_json(no_ec_but_uniprot_acc_ids)
        acc_uniprot_data = {
            acc: acc_uniprot_data[acc]["ecs"] for acc in acc_uniprot_data.keys()
        }

        rhea_reactions_df.loc[mask, "ec"] = (
            rhea_reactions_df.loc[mask, "uniprot_id"]
            .map(acc_uniprot_data)
            .apply(lambda x: normalize_ec_collection(x, fallback=None))
        )
        rhea_reactions_df = rhea_reactions_df.explode("ec")

        rhea_reactions_df["ec"] = rhea_reactions_df["ec"].apply(
            lambda x: normalize_ec_collection(x, fallback=None)
        )
        rhea_reactions_df = rhea_reactions_df.explode("ec").reset_index(drop=True)

        acc_uniprot_data = list(
            get_uniprot_acc_json(
                rhea_reactions_df["uniprot_id"].dropna().unique().tolist()
            ).values()
        )
        acc_uniprot_data = (
            pd.DataFrame(acc_uniprot_data)
            .drop(columns=["ecs"])
            .rename(columns={"acc_id": "uniprot_id"})
        )
        rhea_reactions_df = rhea_reactions_df.merge(
            acc_uniprot_data, on="uniprot_id", how="left"
        )

        self.logger.info("Downloading PubMed abstracts...")
        download_pubmed_abstracts_parallel(
            list(rhea_reactions_df["pubmed_id"].dropna().explode().unique())
        )
        self.logger.info("Parsing all reference XMLs...")
        all_refs = pd.DataFrame(parse_all_reference_xmls()).rename(
            columns={"pmid": "pubmed_id"}
        )

        self.logger.info("Saving Rhea reactions to parquet...")
        try:
            rhea_reactions_df.to_parquet(
                self.RHEA_REACTIONS_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info("Completed.")
        except Exception as e:
            self.logger.error(f"Failed to save Rhea reactions to parquet: {e}")
            return

        self.logger.info("Saving reference data to parquet...")
        try:
            all_refs.to_parquet(
                self.RHEA_REFERENCES_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info("Reference data saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save reference data to parquet: {e}")
            return

        self.logger.info(
            f"No. of unique Rhea reactions: {rhea_reactions_df['rhea_id'].nunique()}"
        )
        self.logger.info(
            f"No. of unique enzyme-catalysed reactions: {rhea_reactions_df[rhea_reactions_df['ec'].notna() & rhea_reactions_df['uniprot_id'].notna()]['rhea_id'].nunique()}"
        )
        self.logger.info(
            f"No. of unique non-enzyme-catalysed reactions: {rhea_reactions_df[rhea_reactions_df['ec'].isna() & rhea_reactions_df['uniprot_id'].isna()]['rhea_id'].nunique()}"
        )
        self.logger.info(
            f"No. of unique reactions with reaction SMILES: {rhea_reactions_df.drop_duplicates(subset='rhea_id')['rxn_smiles'].notna().sum()}"
        )
        self.logger.info(
            f"No. of unique reactions without reaction SMILES: {rhea_reactions_df.drop_duplicates(subset='rhea_id')['rxn_smiles'].isna().sum()}"
        )
        self.logger.info(
            f"No. of unique reactions with PubMed IDs: {rhea_reactions_df.drop_duplicates(subset='rhea_id')['pubmed_id'].notna().sum()}"
        )
        self.logger.info(
            f"No. of unique reactions without PubMed IDs: {rhea_reactions_df.drop_duplicates(subset='rhea_id')['pubmed_id'].isna().sum()}"
        )
        self.logger.info(
            f"No. of PubMed IDs: {rhea_reactions_df['pubmed_id'].nunique()}"
        )
        self.logger.info(f"Total No. of reactions: {rhea_reactions_df.shape[0]}")
        return rhea_reactions_df, all_refs


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = RheaDatasetBuilder(cfg)
    rhea_df, refs_df = builder.setup()
