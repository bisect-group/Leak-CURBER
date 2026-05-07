import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import re
import sexpdata
import numpy as np
import pandas as pd
from rdkit import Chem
from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize, compose

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
smiles_to_mol = chem_utils.smiles_to_mol
get_uniprot_acc_json = chem_utils.get_uniprot_acc_json
normalize_ec_collection = chem_utils.normalize_ec_collection


class MetaCycDatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.ENZRXNS_PATH = Path(cfg.metacyc.metacyc_enzrxns_file_path)
        self.PROTSEQ_PATH = Path(cfg.metacyc.metacyc_protseq_file_path)
        self.PROTEINS_PATH = Path(cfg.metacyc.metacyc_proteins_file_path)
        self.REACTIONS_PATH = Path(cfg.metacyc.metacyc_reactions_file_path)
        self.COMPOUNDS_PATH = Path(cfg.metacyc.metacyc_compounds_file_path)
        self.PROTEIN_SEQ_IDS_PATH = Path(cfg.metacyc.metacyc_protein_seq_ids_file_path)
        self.ATOM_MAPPED_SMILES_PATH = Path(
            cfg.metacyc.metacyc_atom_mapped_smiles_file_path
        )

        self.MOL_PATH = Path(cfg.metacyc.mol_path)
        self.REACTIONS_PARQUET_PATH = Path(
            cfg.metacyc.metacyc_reactions_parquet_file_path
        )
        self.COMPOUNDS_PARQUET_PATH = Path(
            cfg.metacyc.metacyc_compounds_parquet_file_path
        )
        self.KINETIC_PARAMS_PARQUET_PATH = Path(
            cfg.metacyc.metacyc_kinetic_params_parquet_file_path
        )

        LOG_PATH = Path(cfg.metacyc.log_dir)

        for path in [
            LOG_PATH,
            self.REACTIONS_PARQUET_PATH.parent,
            self.COMPOUNDS_PARQUET_PATH.parent,
            self.KINETIC_PARAMS_PARQUET_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.metacyc.log_file_name
        ).get_logger()

        for path in [
            self.ENZRXNS_PATH,
            self.PROTSEQ_PATH,
            self.PROTEINS_PATH,
            self.REACTIONS_PATH,
            self.COMPOUNDS_PATH,
            self.ATOM_MAPPED_SMILES_PATH,
        ]:
            if not path.exists():
                self.logger.error(f"Required MetaCyc file not found: {path}")
                raise FileNotFoundError(f"Required MetaCyc file not found: {path}")

    def parse_attribute_value_file(self, text: str):
        """
        Parses a MetaCyc attribute-value file into a list of entry dictionaries.

        Args:
            text (str): The raw text content of the attribute-value file.

        Returns:
            list[dict]: A list of dictionaries, each representing an entry with keys as attributes.
                        Each attribute is a list of dicts with at least a 'value' key, and optionally 'annotations'.
        """
        entries = []
        current_entry = {}
        last_key = None
        buffer = ""
        annotation_group = {}
        last_is_annotation_group = False

        lines = text.splitlines()

        def flush_buffer():
            nonlocal buffer, last_key, annotation_group, last_is_annotation_group, current_entry
            if buffer and last_key:
                entry = {"value": buffer}
                if last_is_annotation_group and annotation_group:
                    entry["annotations"] = annotation_group.copy()
                current_entry.setdefault(last_key, []).append(entry)
                buffer = ""
                annotation_group = {}
                last_is_annotation_group = False

        for line in lines:
            line = line.rstrip()

            if not line or line.startswith("#"):
                continue  # skip empty or comment lines

            if line.strip() == "//":
                flush_buffer()
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                last_key = None
                continue

            if line.startswith("^"):
                if not last_key:
                    self.logger.warning("Orphan annotation found: %s", line)
                    continue  # orphan annotation
                try:
                    split_line = line[1:].split(" - ", 1)
                    if len(split_line) == 2:
                        ann_key, ann_value = split_line
                    elif len(split_line) == 1:
                        if split_line[0].isupper():
                            ann_key = split_line[0]
                            ann_value = ""
                        else:
                            ann_key = last_key
                            ann_value = split_line[0]
                    else:
                        self.logger.warning(f"Unexpected annotation format: {line}")
                        continue
                except Exception as e:
                    self.logger.error(
                        f"Error processing annotation line: {line} >> {e}"
                    )
                    continue
                annotation_group[ann_key.strip()] = ann_value.strip()
                last_is_annotation_group = True
                continue

            if line.startswith("/"):
                buffer += "\n" + line[1:]
                continue

            # New attribute line
            if " - " in line:
                flush_buffer()
                try:
                    key, value = line.split(" - ", 1)
                    last_key = key.strip()
                    buffer = value.strip()
                    annotation_group = {}
                    last_is_annotation_group = False
                except Exception as e:
                    self.logger.error(f"Error splitting attribute line: {line} >> {e}")
                    continue

        # Add final entry if file doesn't end with //
        flush_buffer()
        if current_entry:
            entries.append(current_entry)

        return entries

    def split_columns_by_annotations(self, df):
        """
        Splits DataFrame columns into those with and without annotation dictionaries.

        Args:
            df (pd.DataFrame): DataFrame where each cell is a list of dicts as produced by parse_attribute_value_file.

        Returns:
            tuple[list[str], list[str]]:
                - cols_with_annotations: columns where at least one cell contains a dict with an 'annotations' key
                - cols_without_annotations: columns where no cell contains a dict with an 'annotations' key
        """
        cols_with_annotations = []
        cols_without_annotations = []
        try:
            for col in df.columns:
                has_annotation = False
                for cell in df[col]:
                    if isinstance(cell, list):
                        for item in cell:
                            if isinstance(item, dict) and "annotations" in item:
                                has_annotation = True
                                break
                    if has_annotation:
                        break
                if has_annotation:
                    cols_with_annotations.append(col)
                else:
                    cols_without_annotations.append(col)
        except Exception as e:
            self.logger.error(f"Error splitting columns by annotations: {e}")
            raise
        return cols_with_annotations, cols_without_annotations

    def strip_annotation_keys(self, df, cols_without_annotations):
        """
        Strips annotation keys from columns that do not contain any annotations.

        Args:
            df (pd.DataFrame): DataFrame to process.
            cols_without_annotations (list[str]): List of column names to strip annotation keys from.

        Returns:
            pd.DataFrame: DataFrame with specified columns' cells simplified to just values.
        """
        df = df.copy()
        for col in cols_without_annotations:

            def strip_cell(cell):
                try:
                    if isinstance(cell, list):
                        # List of dicts or values
                        new_list = []
                        for item in cell:
                            if (
                                isinstance(item, dict)
                                and "value" in item
                                and len(item) == 1
                            ):
                                new_list.append(item["value"])
                            else:
                                new_list.append(item)
                        return new_list
                    elif isinstance(cell, dict) and "value" in cell and len(cell) == 1:
                        return cell["value"]
                    else:
                        return cell
                except Exception as e:
                    self.logger.error(
                        f"Error stripping annotation keys in column '{col}': {e}"
                    )
                    return cell

            try:
                df[col] = df[col].apply(strip_cell)
            except Exception as e:
                self.logger.error(f"Error applying strip_cell to column '{col}': {e}")
                raise
        return df

    def read_fasta(self, file_path):
        """
        Reads a FASTA file and extracts sequence information.

        Args:
            file_path (str or Path): Path to the FASTA file.

        Returns:
            list[tuple]: List of tuples (key, name, organism, sequence).
        """
        sequences = []
        header = name = organism = None
        try:
            with open(file_path, "r") as file:
                header = key = name = sequence = None
                for line in file.read().strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        header = line[1:].split("|")[-1].split(" ")
                        key = header[0]
                        header = " ".join(header[1:])
                        name = header.split(" (")[0]
                        organism = (
                            header.split(" (")[1].rstrip(")") if "(" in header else None
                        )

                    else:  # Sequence line
                        sequences.append((key, name, organism, line))
            self.logger.info(
                f"Successfully read {len(sequences)} sequences from {file_path}"
            )
        except Exception as e:
            self.logger.error(f"Error reading FASTA file {file_path}: {e}")
            raise
        return sequences

    def clean_EC_NUM(self, ec_num):
        """
        Cleans and extracts the EC number from a MetaCyc EC-NUM field.

        Args:
            ec_num (list): List of EC number dicts from MetaCyc.

        Returns:
            list[str | None]: Normalized EC token list.
        """
        try:
            if not isinstance(ec_num, list) or len(ec_num) == 0:
                return [None]

            candidates = []
            for entry in ec_num:
                if isinstance(entry, dict) and "value" in entry:
                    candidates.append(str(entry["value"]).replace("EC-", ""))
                elif isinstance(entry, str):
                    candidates.append(entry.replace("EC-", ""))

            if not candidates:
                return [None]

            return normalize_ec_collection(candidates, fallback=None)
        except Exception as e:
            self.logger.error(f"Error cleaning EC_NUM: {e}")
            return [None]

    def clean_DBLINKS(self, dblinks):
        """
        Parse a DBLINKS cell (list of strings or string) and extract (db, id) tuples.

        Args:
            dblinks (list or str): DBLINKS cell from MetaCyc.

        Returns:
            tuple of tuples: [(db, id), ...]
        """
        try:
            if dblinks is None or (isinstance(dblinks, float) and np.isnan(dblinks)):
                return
            result = []
            pattern = re.compile(r'\(\s*([A-Z0-9\-]+)\s+"([^"]+)"')
            for entry in dblinks:
                matches = pattern.findall(entry)
                for db, id_ in matches:
                    result.append((db, id_))
            return result
        except Exception as e:
            self.logger.error(f"Error cleaning DBLINKS: {e}")
            return None

    def clean_reaction_side(self, participants):
        """
        Cleans the reaction side participants from MetaCyc.

        Args:
            participants (list): List of participant dicts.

        Returns:
            list: List of (coefficient, compound) tuples.
        """
        try:
            if not isinstance(participants, list):
                return None
            cleaned_side = []
            for p in participants:
                val = p["value"]
                coeff = p.get("annotations", {}).get("COEFFICIENT", 1)
                cleaned_side.append((str(coeff), str(val)))
            return cleaned_side
        except Exception as e:
            self.logger.error(f"Error cleaning reaction side: {e}")
            return None

    def generate_reaction_smiles(self, row, compound_smiles_mapping):
        """
        Generates atom-mapped reaction SMILES for a reaction row.

        Args:
            row (pd.Series): Row containing LEFT, RIGHT, and ATOM-MAPPED-SMILES fields.
            compound_smiles_mapping (dict): Mapping from compound IDs to SMILES.

        Returns:
            pd.Series: Updated row with ATOM-MAPPED-SMILES field.
        """
        try:
            if row["ATOM-MAPPED-SMILES"]:
                return row

            left, right = row["LEFT"], row["RIGHT"]
            if not left or not right:
                row["ATOM-MAPPED-SMILES"] = None
                return row

            left_smiles, right_smiles = [], []

            for coeff, compound in left:
                try:
                    coeff_int = int(coeff)
                except Exception:
                    coeff_int = 1
                left_smiles.extend(
                    [compound_smiles_mapping.get(compound, None)] * coeff_int
                )

            for coeff, compound in right:
                try:
                    coeff_int = int(coeff)
                except Exception:
                    coeff_int = 1
                right_smiles.extend(
                    [compound_smiles_mapping.get(compound, None)] * coeff_int
                )

            row["ATOM-MAPPED-SMILES"] = (
                f"{'.'.join(left_smiles)}>>{'.'.join(right_smiles)}"
                if None not in left_smiles + right_smiles
                else None
            )
            return row
        except Exception as e:
            self.logger.error(f"Error generating reaction SMILES: {e}")
            row["ATOM-MAPPED-SMILES"] = None
            return row

    def expand_kinetic_df(self, row, type):
        """
        Expands a kinetic parameter dictionary into separate columns.

        Args:
            row (pd.Series): Row containing kinetic parameter dict.
            type (str): The kinetic parameter type (e.g., 'KCAT').

        Returns:
            pd.Series: Updated row with expanded fields.
        """
        try:
            kcat_dict = row[type]
            val = kcat_dict["value"]
            try:
                substrate = kcat_dict["annotations"]["SUBSTRATE"]
            except KeyError:
                substrate = None
            try:
                cite = kcat_dict["annotations"]["CITATIONS"]
            except KeyError:
                cite = None
            row[type] = val
            row["SUBSTRATE"] = substrate
            row["CITATIONS"] = cite
            return row
        except Exception as e:
            self.logger.error(f"Error expanding kinetic df for type {type}: {e}")
            return row

    def sexp_to_str(self, obj):
        """
        Recursively converts S-expression objects to strings.

        Args:
            obj: S-expression object.

        Returns:
            str or list: Converted string or list of strings.
        """
        try:
            if isinstance(obj, list):
                return [self.sexp_to_str(x) for x in obj]
            elif isinstance(obj, sexpdata.Symbol):
                return str(obj)
            else:
                return obj
        except Exception as e:
            self.logger.error(f"Error converting S-expression to string: {e}")
            return obj

    def try_alternate_accs(self, row, results):
        """
        Tries alternate sequence accession IDs to find a valid sequence.

        Args:
            row (pd.Series): Row with SEQ-ID and PRIM-SEQ-ID fields.
            results (dict): Mapping of accession IDs to sequence data.

        Returns:
            pd.Series: Updated row with valid PRIM-SEQ-ID and SEQUENCE if found.
        """
        try:
            if not isinstance(row["SEQ-ID"], list):
                return row
            for acc in row["SEQ-ID"]:
                if acc == row["PRIM-SEQ-ID"]:
                    continue
                seq_data = results.get(acc)
                if seq_data is None:
                    # Try to download if not already in results
                    seq_data = get_uniprot_acc_json([acc]).get(acc)
                    results[acc] = seq_data
                if seq_data and seq_data.get("sequence"):
                    row["PRIM-SEQ-ID"] = acc
                    row["SEQUENCE"] = seq_data["sequence"]
                    return row
            return row
        except Exception as e:
            self.logger.error(f"Error trying alternate accession IDs: {e}")
            return row

    def parse_compounds(self):
        """
        Parses the MetaCyc compounds.dat file and returns a DataFrame of compounds and a mapping from compound IDs to SMILES.

        Returns:
            tuple: (compounds_df, compound_smiles_mapping)
                - compounds_df (pd.DataFrame): DataFrame containing compound information.
                - compound_smiles_mapping (dict): Mapping from compound UNIQUE-ID to SMILES string.
        """
        try:
            self.logger.info(f"Reading compounds from {self.COMPOUNDS_PATH}")
            with open(self.COMPOUNDS_PATH, "r", encoding="latin-1") as f:
                compounds_df = pd.DataFrame(self.parse_attribute_value_file(f.read()))
            compounds_df.drop(
                columns=["CREDITS", "INSTANCE-NAME-TEMPLATE", "SYNONYMS", "COMMENT"],
                inplace=True,
            )
            self.logger.info("Splitting columns by annotations")
            (
                cols_with_annotations,
                cols_without_annotations,
            ) = self.split_columns_by_annotations(compounds_df)
            compounds_df = self.strip_annotation_keys(
                compounds_df, cols_without_annotations
            )
            compounds_df = compounds_df[
                [
                    "UNIQUE-ID",
                    "TYPES",
                    "COMMON-NAME",
                    "INCHI",
                    "SMILES",
                    "DBLINKS",
                    "CHEMICAL-FORMULA",
                    "GIBBS-0",
                    "MOLECULAR-WEIGHT",
                    "MONOISOTOPIC-MW",
                    "CITATIONS",
                ]
            ]
            self.logger.info("Cleaning CHEMICAL-FORMULA column")
            compounds_df["CHEMICAL-FORMULA"] = compounds_df["CHEMICAL-FORMULA"].apply(
                lambda x: (
                    "".join(
                        f"{m.group(1)}{m.group(2)}"
                        for i in x
                        if (m := re.match(r"\((\w+)\s+(\d+)\)", i))
                    )
                    if isinstance(x, list)
                    else x
                )
            )
            for col in (
                "UNIQUE-ID",
                "COMMON-NAME",
                "INCHI",
                "SMILES",
                "GIBBS-0",
                "MOLECULAR-WEIGHT",
                "MONOISOTOPIC-MW",
            ):
                compounds_df[col] = compounds_df[col].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )

            self.logger.info(
                "Validating SMILES strings and deleting invalid ones or empty strings"
            )
            compounds_df["SMILES"] = compounds_df["SMILES"].apply(
                lambda x: (
                    x if pd.notna(x) and Chem.MolFromSmiles(x) is not None else None
                )
            )
            compounds_df = compounds_df.dropna(subset=["SMILES"]).reset_index(drop=True)
            compound_smiles_mapping = dict(
                zip(compounds_df["UNIQUE-ID"], compounds_df["SMILES"])
            )
            compound_formula_mapping = dict(
                zip(compounds_df["UNIQUE-ID"], compounds_df["CHEMICAL-FORMULA"])
            )
            self.logger.info(
                f"Parsed {len(compounds_df)} compounds with valid SMILES from {self.COMPOUNDS_PATH}"
            )
            self.logger.info(
                f"Generating MOL files for {len(compound_smiles_mapping)} compounds"
            )
            smiles_to_mol(compound_smiles_mapping)

            return compounds_df, compound_smiles_mapping, compound_formula_mapping
        except Exception as e:
            self.logger.error(f"Error parsing compounds: {e}")
            raise

    def augment_rxn(self, row, compound_formula_mapping):
        """
        Augments a reaction row with human-readable reaction definitions and equations, and extracts external database IDs.

        This function generates two new fields for the reaction:
            - 'DEFINITION': A string representing the reaction in terms of compound names and stoichiometry.
            - 'EQUATION': A string representing the reaction in terms of compound formulas and stoichiometry.

        It also parses the 'DBLINKS' field to extract and assign lists of external database IDs:
            - 'kegg_id': KEGG reaction IDs (from LIGAND-RXN)
            - 'metanetx_id': MetaNetX reaction IDs (from METANETX-RXN)
            - 'rhea_id': Rhea reaction IDs (from RHEA)

        Args:
            row (pd.Series): A row from the reactions DataFrame, containing at least the columns 'LEFT', 'RIGHT', and 'DBLINKS'.
            compound_formula_mapping (dict): Mapping from compound IDs to their chemical formulas.

        Returns:
            pd.Series: The input row, augmented with 'DEFINITION', 'EQUATION', and external database ID fields.
        """

        def generate_side_eqn(side, compound_formula_mapping):
            """
            Generate a side equation string for a given side (substrates or products) from a list of (coeff, name) tuples.
            """
            try:
                definition = " + ".join([" ".join(elem) for elem in side])
            except:
                self.logger.error(
                    f"Error generating side equation for row {row.name}: {side}"
                )
                definition = None

            try:
                equation = " + ".join(
                    [f"{elem[0]} {compound_formula_mapping[elem[1]]}" for elem in side]
                )
            except KeyError as e:
                self.logger.error(f"Missing compound formula for {e} in row {row.name}")
                equation = None
            except:
                self.logger.error(
                    f"Error generating equation for row {row.name}: {side}"
                )
                equation = None

            return definition, equation

        # Use pd.isna to check for missing values robustly, handling arrays/lists

        left_missing = (
            row["LEFT"] is None
            or (hasattr(row["LEFT"], "__len__") and pd.isna(row["LEFT"]).any())
            if hasattr(row["LEFT"], "__len__")
            else pd.isna(row["LEFT"])
        )
        right_missing = (
            row["RIGHT"] is None
            or (hasattr(row["RIGHT"], "__len__") and pd.isna(row["RIGHT"]).any())
            if hasattr(row["RIGHT"], "__len__")
            else pd.isna(row["RIGHT"])
        )

        if left_missing or right_missing:
            row["DEFINITION"] = None
            row["EQUATION"] = None
        else:
            left_definition, left_equation = generate_side_eqn(
                row["LEFT"], compound_formula_mapping
            )
            right_definition, right_equation = generate_side_eqn(
                row["RIGHT"], compound_formula_mapping
            )
            row["DEFINITION"] = (
                f"{left_definition} = {right_definition}"
                if left_definition and right_definition
                else None
            )
            row["EQUATION"] = (
                f"{left_equation} = {right_equation}"
                if left_equation and right_equation
                else None
            )

        dblinks = row["DBLINKS"]
        if not isinstance(dblinks, list):
            row["kegg_id"] = None
            row["metanetx_id"] = None
            row["rhea_id"] = None
        else:
            for db, id_ in dblinks:
                if db == "LIGAND-RXN":
                    try:
                        row["kegg_id"].append(id_)
                    except KeyError:
                        row["kegg_id"] = [id_]
                elif db == "METANETX-RXN":
                    try:
                        row["metanetx_id"].append(id_)
                    except KeyError:
                        row["metanetx_id"] = [id_]
                elif db == "RHEA":
                    try:
                        row["rhea_id"].append(f"RHEA:{id_}")
                    except KeyError:
                        row["rhea_id"] = [f"RHEA:{id_}"]
        return row

    def parse_reactions(self, compound_smiles_mapping, compound_formula_mapping):
        """
        Parses the MetaCyc reactions.dat and atom-mappings-smiles.dat files, processes reaction information,
        and returns a DataFrame of reactions and a mapping from enzymatic reactions to reactions.

        Args:
            compound_smiles_mapping (dict): Mapping from compound IDs to SMILES.

        Returns:
            tuple: (reactions_df, enz_2_rxn_map)
                - reactions_df (pd.DataFrame): DataFrame containing reaction information.
                - enz_2_rxn_map (pd.DataFrame): Mapping from enzymatic reaction IDs to reaction IDs and EC numbers.
        """
        try:
            self.logger.info(
                f"Reading atom-mapped SMILES from {self.ATOM_MAPPED_SMILES_PATH}"
            )
            with open(self.ATOM_MAPPED_SMILES_PATH, "r", encoding="latin-1") as f:
                atom_mapped_smiles_df = pd.DataFrame(
                    [line.split("\t") for line in f.readlines()],
                    columns=["UNIQUE-ID", "ATOM-MAPPED-SMILES"],
                )

            self.logger.info(f"Reading reactions from {self.REACTIONS_PATH}")
            with open(self.REACTIONS_PATH, "r", encoding="latin-1") as f:
                reactions_df = pd.DataFrame(self.parse_attribute_value_file(f.read()))
            reactions_df.drop(
                columns=[
                    "ATOM-MAPPINGS",
                    "CREDITS",
                    "INSTANCE-NAME-TEMPLATE",
                    "MEMBER-SORT-FN",
                    "ENZYMES-NOT-USED",
                    "REACTION-LIST",
                    "RXN-LOCATIONS",
                    "STD-REDUCTION-POTENTIAL",
                    "PREDECESSORS",
                    "PRIMARIES",
                    "SYNONYMS",
                    "REGULATED-BY",
                    "COMMENT",
                    "SIGNAL",
                    "TYPES",
                    "IN-PATHWAY",
                    "ORPHAN?",
                    "PHYSIOLOGICALLY-RELEVANT?",
                    "COMMON-NAME",
                    "CANNOT-BALANCE?",
                    "SYSTEMATIC-NAME",
                ],
                inplace=True,
            )

            for col in [
                "UNIQUE-ID",
                "REACTION-BALANCE-STATUS",
                "REACTION-DIRECTION",
                "GIBBS-0",
                "SPONTANEOUS?",
            ]:
                reactions_df[col] = reactions_df[col].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )

            self.logger.info("Splitting columns by annotations")
            (
                cols_with_annotations,
                cols_without_annotations,
            ) = self.split_columns_by_annotations(reactions_df)
            reactions_df = self.strip_annotation_keys(
                reactions_df, cols_without_annotations
            )

            self.logger.info("Cleaning EC-NUMBER and DBLINKS columns")
            reactions_df["EC-NUMBER"] = reactions_df["EC-NUMBER"].apply(
                self.clean_EC_NUM
            )
            reactions_df = reactions_df.explode("EC-NUMBER").reset_index(drop=True)
            reactions_df["DBLINKS"] = reactions_df["DBLINKS"].apply(self.clean_DBLINKS)

            self.logger.info("Cleaning LEFT and RIGHT reaction sides")
            reactions_df["LEFT"] = reactions_df["LEFT"].apply(self.clean_reaction_side)
            reactions_df["RIGHT"] = reactions_df["RIGHT"].apply(
                self.clean_reaction_side
            )
            reactions_df = reactions_df.merge(
                atom_mapped_smiles_df, on="UNIQUE-ID", how="left"
            )

            self.logger.info("Generating reaction SMILES")
            reactions_df = reactions_df.apply(
                lambda row: self.generate_reaction_smiles(row, compound_smiles_mapping),
                axis=1,
            )

            # Generate reaction definitions
            self.logger.info(
                "Generating reaction definitions & equations and parsing database cross-references"
            )
            reactions_df = reactions_df.apply(
                lambda row: self.augment_rxn(row, compound_formula_mapping), axis=1
            ).drop(columns=["LEFT", "RIGHT", "DBLINKS"])

            self.logger.info("Building enzymatic reaction to reaction mapping")
            enz_2_rxn_map = (
                reactions_df[reactions_df["ENZYMATIC-REACTION"].notna()][
                    ["UNIQUE-ID", "ENZYMATIC-REACTION", "EC-NUMBER"]
                ]
                .explode("ENZYMATIC-REACTION")
                .explode("EC-NUMBER")
                .reset_index(drop=True)
                .rename(
                    columns={"UNIQUE-ID": "RXN-ID", "ENZYMATIC-REACTION": "ENZRXN-ID"}
                )
            )

            self.logger.info(
                f"Parsed {len(reactions_df)} reactions and {len(enz_2_rxn_map)} enzymatic reaction mappings"
            )
            return reactions_df, enz_2_rxn_map
        except Exception as e:
            self.logger.error(f"Error parsing reactions: {e}")
            raise

    def parse_proteins(self, enz_2_rxn_map):
        """
        Parses MetaCyc protein data and returns merged protein and sequence DataFrames.

        Args:
            enz_2_rxn_map (pd.DataFrame): Mapping from enzymatic reaction IDs to reaction IDs and EC numbers.

        Returns:
            tuple: (proteins_df, protein_seq_ids_df)
                - proteins_df (pd.DataFrame): DataFrame containing protein information merged with sequence and reaction mapping.
                - protein_seq_ids_df (pd.DataFrame): DataFrame mapping RXN-ID and EC-NUMBER to UniProt sequence IDs and sequences.
        """
        try:
            self.logger.info(f"Reading protein sequences from {self.PROTSEQ_PATH}")
            protseq_df = pd.DataFrame(
                self.read_fasta(self.PROTSEQ_PATH),
                columns=["UNIQUE-ID", "NAME", "ORGANISM", "SEQUENCE"],
            )

            self.logger.info(f"Reading proteins from {self.PROTEINS_PATH}")
            with open(self.PROTEINS_PATH, "r", encoding="latin-1") as f:
                proteins_df = pd.DataFrame(self.parse_attribute_value_file(f.read()))
            proteins_df = proteins_df[
                [
                    "UNIQUE-ID",
                    "CATALYZES",
                    "CITATIONS",
                    "SPECIES",
                    "DBLINKS",
                    "GENE",
                    "SYNONYMS",
                ]
            ]

            self.logger.info("Splitting protein columns by annotations")
            cols_with_annotations, cols_without_annotations = (
                self.split_columns_by_annotations(proteins_df)
            )
            proteins_df = self.strip_annotation_keys(
                proteins_df, cols_without_annotations
            )

            for col in ["UNIQUE-ID", "SPECIES"]:
                proteins_df[col] = proteins_df[col].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )

            self.logger.info("Cleaning protein DBLINKS")
            proteins_df["DBLINKS"] = proteins_df["DBLINKS"].apply(self.clean_DBLINKS)

            self.logger.info(
                "Merging protein DataFrame with sequence and reaction mapping"
            )
            proteins_df = (
                proteins_df.merge(protseq_df, on="UNIQUE-ID", how="inner")
                .explode("CATALYZES")
                .dropna(subset=["CATALYZES"])
                .merge(
                    enz_2_rxn_map.rename(columns={"ENZRXN-ID": "CATALYZES"}),
                    on="CATALYZES",
                    how="inner",
                )
                .drop(columns=["NAME", "SYNONYMS", "GENE", "SPECIES"])
            )

            self.logger.info("Cleaning protein DBLINKS to keep only UniProt entries")
            proteins_df["DBLINKS"] = proteins_df["DBLINKS"].apply(
                lambda x: (
                    [elem[1] for elem in x if elem[0] == "UNIPROT"][0]
                    if isinstance(x, list)
                    else x
                )
            )

            self.logger.info(
                f"Reading protein sequence IDs from {self.PROTEIN_SEQ_IDS_PATH}"
            )
            with open(self.PROTEIN_SEQ_IDS_PATH, "r", encoding="latin-1") as f:
                protein_seq_ids = " ".join(
                    [
                        line.strip()
                        for line in f.readlines()
                        if line.strip() and not line.strip().startswith(";;")
                    ]
                )

            protein_seq_ids = [
                {
                    "RXN-ID": row[0],
                    "EC-NUMBER": row[1],
                    "SEQ-ID": sorted(
                        [prot[8:] for prot in row[2:] if prot.startswith("UNIPROT:")],
                        key=len,
                    ),
                }
                for row in self.sexp_to_str(sexpdata.loads(protein_seq_ids))
            ]
            protein_seq_ids_df = pd.DataFrame(protein_seq_ids)
            protein_seq_ids_df["EC-NUMBER"] = protein_seq_ids_df["EC-NUMBER"].apply(
                lambda x: normalize_ec_collection(x, fallback=None)
            )
            protein_seq_ids_df = protein_seq_ids_df.explode("EC-NUMBER").reset_index(
                drop=True
            )
            protein_seq_ids_df = protein_seq_ids_df[
                ~(
                    protein_seq_ids_df["SEQ-ID"].apply(
                        lambda x: isinstance(x, list) and len(x) == 0
                    )
                )
            ].reset_index(drop=True)
            protein_seq_ids_df["PRIM-SEQ-ID"] = protein_seq_ids_df["SEQ-ID"].apply(
                lambda x: x[0] if isinstance(x, list) else None
            )

            self.logger.info("Fetching UniProt sequences for primary sequence IDs")
            primary_results = get_uniprot_acc_json(
                protein_seq_ids_df["PRIM-SEQ-ID"].dropna().unique()
            )
            protein_seq_ids_df["SEQUENCE"] = protein_seq_ids_df["PRIM-SEQ-ID"].map(
                lambda acc: primary_results.get(acc, {}).get("sequence", None)
            )

            self.logger.info("Trying alternate accession IDs for missing sequences")
            missing_seq_mask = protein_seq_ids_df["SEQUENCE"].isna()
            protein_seq_ids_df.loc[missing_seq_mask] = protein_seq_ids_df.loc[
                missing_seq_mask
            ].apply(lambda row: self.try_alternate_accs(row, primary_results), axis=1)
            protein_seq_ids_df = (
                protein_seq_ids_df.drop_duplicates(
                    subset=["RXN-ID", "PRIM-SEQ-ID", "SEQUENCE"]
                )
                .dropna(subset=["SEQUENCE"])
                .reset_index(drop=True)
            )

            self.logger.info(
                f"Parsed {len(proteins_df)} proteins and {len(protein_seq_ids_df)} protein sequence IDs"
            )
            return proteins_df, protein_seq_ids_df
        except Exception as e:
            self.logger.error(f"Error parsing proteins: {e}")
            raise

    def parse_enzrxns(self, proteins_df):
        """
        Parses the MetaCyc enzrxns.dat file and merges with protein DataFrame.

        Args:
            proteins_df (pd.DataFrame): DataFrame containing protein information.

        Returns:
            pd.DataFrame: DataFrame containing enzymatic reactions merged with protein info.
        """
        try:
            with open(self.ENZRXNS_PATH, "r", encoding="latin-1") as f:
                self.logger.info(
                    f"Reading enzymatic reactions from {self.ENZRXNS_PATH}"
                )
                enzrxns_df = pd.DataFrame(self.parse_attribute_value_file(f.read()))
            self.logger.info("Splitting enzymatic reaction columns by annotations")
            cols_with_annotations, cols_without_annotations = (
                self.split_columns_by_annotations(enzrxns_df)
            )
            enzrxns_df = self.strip_annotation_keys(
                enzrxns_df, cols_without_annotations
            )
            enzrxns_df.drop(
                columns=[
                    "TYPES",
                    "INSTANCE-NAME-TEMPLATE",
                    "CREDITS",
                    "PHYSIOLOGICALLY-RELEVANT?",
                    "COMMENT",
                    "SYNONYMS",
                    "SOURCE-ORTHOLOG",
                    "REQUIRED-PROTEIN-COMPLEX",
                    "COFACTOR-BINDING-COMMENT",
                    "ALTERNATIVE-COFACTORS",
                    "ENZRXN-IN-PATHWAY",
                    "BASIS-FOR-ASSIGNMENT",
                ],
                inplace=True,
            )
            for col in [
                "UNIQUE-ID",
                "COMMON-NAME",
                "ENZYME",
                "REACTION",
                "REACTION-DIRECTION",
            ]:
                enzrxns_df[col] = enzrxns_df[col].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )

            self.logger.info("Merging enzymatic reactions with protein DataFrame")
            enzrxns_df = enzrxns_df.merge(
                proteins_df[["UNIQUE-ID", "ORGANISM", "SEQUENCE"]].rename(
                    columns={"UNIQUE-ID": "ENZYME"}
                ),
                how="left",
            )
            self.logger.info(
                f"Parsed {len(enzrxns_df)} enzymatic reactions from {self.ENZRXNS_PATH}"
            )
            return enzrxns_df
        except Exception as e:
            self.logger.error(f"Error parsing enzymatic reactions: {e}")
            raise

    def process_kinetic_df(
        self, enzrxns_df, kinetic_type, enz_2_rxn_map, compounds_df, protein_seq_ids_df
    ):
        """
        Processes kinetic parameter data (e.g., KCAT, KM) from enzymatic reactions DataFrame.

        Args:
            enzrxns_df (pd.DataFrame): DataFrame of enzymatic reactions.
            kinetic_type (str): The kinetic parameter type (e.g., 'KCAT', 'KM').
            enz_2_rxn_map (pd.DataFrame): Mapping from enzymatic reaction IDs to reaction IDs and EC numbers.
            compounds_df (pd.DataFrame): DataFrame containing compound information.
            protein_seq_ids_df (pd.DataFrame): DataFrame mapping RXN-ID and EC-NUMBER to UniProt sequence IDs and sequences.

        Returns:
            pd.DataFrame: Processed DataFrame with kinetic parameters, substrate SMILES, and sequence info.
        """
        try:
            self.logger.info(
                f"Filtering enzymatic reactions for kinetic type '{kinetic_type}'"
            )
            df = (
                enzrxns_df[enzrxns_df[kinetic_type].notna()][
                    ["UNIQUE-ID", "ENZYME", kinetic_type, "ORGANISM", "SEQUENCE"]
                ]
                .explode(kinetic_type)
                .reset_index(drop=True)
            )

            df = df.apply(lambda row: self.expand_kinetic_df(row, kinetic_type), axis=1)
            df[kinetic_type] = df[kinetic_type].astype(float)
            df = df.dropna(subset=["SUBSTRATE", kinetic_type]).rename(
                columns={"UNIQUE-ID": "ENZRXN-ID"}
            )
            df["SUBSTRATE"] = df["SUBSTRATE"].apply(lambda x: x.strip("|"))

            self.logger.info("Merging with enzymatic reaction to reaction mapping")
            df = (
                df.merge(enz_2_rxn_map, on="ENZRXN-ID", how="left")
                .merge(
                    compounds_df[["UNIQUE-ID", "SMILES"]].rename(
                        columns={"UNIQUE-ID": "SUBSTRATE"}
                    ),
                    on="SUBSTRATE",
                    how="left",
                )
                .dropna(subset=["SMILES"])
                .merge(
                    protein_seq_ids_df.drop(columns=["SEQ-ID"]).rename(
                        columns={"SEQUENCE": "SEQ_MERGED"}
                    ),
                    on=["RXN-ID", "EC-NUMBER"],
                    how="left",
                )
            )

            df["SEQUENCE"] = df["SEQUENCE"].combine_first(df["SEQ_MERGED"])
            df = (
                df.drop(columns=["SEQ_MERGED"])
                .dropna(subset=["SEQUENCE"])
                .reset_index(drop=True)
            )

            mask = df["PRIM-SEQ-ID"].isna()
            if mask.any():
                self.logger.info(
                    f"Filling {mask.sum()} missing PRIM-SEQ-ID values with ENZYME|ORGANISM fallback"
                )
                df.loc[mask, "PRIM-SEQ-ID"] = (
                    df.loc[mask, "ENZYME"].astype(str)
                    + "|"
                    + df.loc[mask, "ORGANISM"].astype(str)
                )

            self.logger.info(
                f"Processed kinetic DataFrame for type '{kinetic_type}' with {len(df)} entries"
            )
            return df
        except Exception as e:
            self.logger.error(
                f"Error processing kinetic DataFrame for type '{kinetic_type}': {e}"
            )
            raise

    def setup(self):
        try:
            self.logger.info("Starting MetaCyc dataset generation")

            # Parse compounds
            compounds_df, compound_smiles_mapping, compound_formula_mapping = (
                self.parse_compounds()
            )

            # Parse reactions
            reactions_df, enz_2_rxn_map = self.parse_reactions(
                compound_smiles_mapping=compound_smiles_mapping,
                compound_formula_mapping=compound_formula_mapping,
            )

            # Parse proteins and protein sequences
            proteins_df, protein_seq_ids_df = self.parse_proteins(enz_2_rxn_map)

            # Add protein information to reactions
            self.logger.info("Adding protein information to reactions")
            reactions_df = reactions_df.merge(
                proteins_df.rename(
                    columns={
                        "UNIQUE-ID": "PROTEIN-ID",
                        "RXN-ID": "UNIQUE-ID",
                        "CATALYZES": "ENZRXN-ID",
                        "CITATIONS": "PROTEIN-CITATIONS",
                        "DBLINKS": "UNIPROT-ID",
                    }
                ).drop(
                    columns=[
                        "EC-NUMBER",
                        "PROTEIN-ID",
                        "PROTEIN-CITATIONS",
                    ]
                ),
                on="UNIQUE-ID",
                how="left",
            ).drop(columns=["ENZYMATIC-REACTION"])

            # Parse enzymatic reactions
            enzrxns_df = self.parse_enzrxns(proteins_df)

            # Process kinetic parameters
            kcat_df = self.process_kinetic_df(
                enzrxns_df, "KCAT", enz_2_rxn_map, compounds_df, protein_seq_ids_df
            ).rename(columns={"KCAT": "value"})
            kcat_df["value_type"] = "kcat"
            kcat_df["unit"] = "s^(-1)"

            km_df = self.process_kinetic_df(
                enzrxns_df, "KM", enz_2_rxn_map, compounds_df, protein_seq_ids_df
            ).rename(columns={"KM": "value"})
            km_df["value_type"] = "km"
            km_df["unit"] = "millimolar (mM)"
            km_df["value"] = km_df["value"] * 1e-3  # Convert from uM to mM

            kinetic_params_df = pd.concat([kcat_df, km_df], ignore_index=True).rename(
                columns={
                    "ORGANISM": "Organism",
                    "SUBSTRATE": "substrate",
                    "SMILES": "smiles",
                    "PRIM-SEQ-ID": "UniprotID",
                    "SEQUENCE": "sequence",
                    "EC-NUMBER": "ECNumber",
                }
            )
            kinetic_params_df["substrate_id"] = kinetic_params_df["substrate"]
        except Exception as e:
            self.logger.error(f"Error in MetaCyc dataset generation: {e}")

        # Save the processed DataFrames to parquet files
        self.logger.info("Saving processed DataFrames to parquet files")
        try:
            reactions_df.to_parquet(
                self.REACTIONS_PARQUET_PATH,
                index=False,
                compression="brotli",
            )
        except Exception as e:
            self.logger.error(f"Error saving metacyc_reactions.parquet: {e}")
        try:
            compounds_df.to_parquet(
                self.COMPOUNDS_PARQUET_PATH,
                index=False,
                compression="brotli",
            )
        except Exception as e:
            self.logger.error(f"Error saving metacyc_compounds.parquet: {e}")
        try:
            kinetic_params_df.to_parquet(
                self.KINETIC_PARAMS_PARQUET_PATH,
                index=False,
                compression="brotli",
            )
        except Exception as e:
            self.logger.error(f"Error saving metacyc_kinetic_params.parquet: {e}")
        self.logger.info("MetaCyc dataset generation completed successfully")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = MetaCycDatasetBuilder(cfg)
    builder.setup()
