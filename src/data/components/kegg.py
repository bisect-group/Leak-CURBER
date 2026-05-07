import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import re
import requests
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from molvs import standardize_smiles
from hydra import compose, initialize
from multiprocessing.pool import ThreadPool
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import MolFromMolFile, MolToSmiles, MolFromSmiles, MolToMolFile


from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
load_json = chem_utils.load_json
save_json = chem_utils.save_json
get_mol_kegg = chem_utils.get_mol_kegg
parse_equation = chem_utils.parse_equation
get_pubchem_compound = chem_utils.get_pubchem_compound
generate_reaction_SMILES = chem_utils.generate_reaction_SMILES
normalize_ec_collection = chem_utils.normalize_ec_collection


class KEGGDatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.KEGG_REACTION_LIST_URL = cfg.kegg.kegg_reaction_list_url

        self.KEGG_MOL_PATH = Path(cfg.kegg.kegg_mol_path)
        self.KEGG_DATA_PATH = Path(cfg.kegg.kegg_data_path)
        self.KEGG_GLYCAN_PATH = Path(cfg.kegg.kegg_glycan_path)
        self.KEGG_COMPOUND_PATH = Path(cfg.kegg.kegg_compound_path)
        self.KEGG_REACTION_PATH = Path(cfg.kegg.kegg_reaction_path)

        self.KEGG_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.kegg.kegg_reactions_parquet_file_path
        )
        self.KEGG_COMPOUNDS_PARQUET_FILE_PATH = Path(
            cfg.kegg.kegg_compounds_parquet_file_path
        )

        LOG_PATH = Path(cfg.kegg.log_dir)

        for path in [
            LOG_PATH,
            self.KEGG_MOL_PATH,
            self.KEGG_DATA_PATH,
            self.KEGG_GLYCAN_PATH,
            self.KEGG_COMPOUND_PATH,
            self.KEGG_REACTION_PATH,
            self.KEGG_REACTIONS_PARQUET_FILE_PATH.parent,
            self.KEGG_COMPOUNDS_PARQUET_FILE_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.kegg.log_file_name
        ).get_logger()

    def parse_kegg_reaction(self, reaction):
        def extract_value(pattern, text):
            try:
                return re.split(pattern, text, 1)[1].strip().split("\n", 1)[0].strip()
            except:
                return None

        def extract_references(text):
            reference_pattern = re.compile(
                r"REFERENCE\s+\d+.*?(?=REFERENCE|\Z)", re.DOTALL
            )
            pmid_pattern = re.compile(r"\[PMID:(\d+)\]")
            doi_pattern = re.compile(r"DOI:([^\s]+)")
            pmids, dois = [], []
            references = reference_pattern.findall(text)
            for ref in references:
                pmid = pmid_pattern.search(ref)
                doi = doi_pattern.search(ref)
                pmids.append(pmid.group(1) if pmid else None)
                dois.append(doi.group(1) if doi else None)
            return pmids or None, dois or None

        equation = extract_value(r"EQUATION\s+([\s\S]+?)(?=\n[A-Z]|$)", reaction)
        substrates, direction, products = parse_equation(equation, r"[CGD]\d{5}")
        pmids, dois = extract_references(reaction)
        ec = extract_value(r"ENZYME\s+([\s\S]+?)(?=\n[A-Z]|$)", reaction)
        ec = normalize_ec_collection(ec, fallback=None)
        rhea_ids = extract_value(r"DBLINKS\s+([\s\S]+?)(?=\n[A-Z]|$)", reaction)
        rhea_ids = [f"RHEA:{r}" for r in rhea_ids[6:].split(" ")] if rhea_ids else None

        return {
            "kegg_id": extract_value(r"ENTRY\s+(R\d+)", reaction),
            "rhea_id": rhea_ids,
            "definition": extract_value(
                r"DEFINITION\s+([\s\S]+?)(?=\n[A-Z]|$)", reaction
            ),
            "substrates": substrates,
            "direction": "REVERSIBLE" if direction == "<=>" else direction,
            "products": products,
            "ec": ec,
            "pmids": pmids,
            "dois": dois,
        }

    def parse_kegg_compound(self, compound):
        def extract_value(pattern, text):
            try:
                return re.split(
                    r"(?:;\s*|\n\s+)", re.split(pattern, text, 1)[1].strip()
                )
            except:
                return None

        kegg_id = extract_value(r"ENTRY\s+(\S+)", compound)[0]
        name = extract_value(r"NAME\s+([\s\S]+?)(?=\n[A-Z]|$)", compound)
        formula = extract_value(r"FORMULA\s+([\s\S]+?)(?=\n[A-Z]|$)", compound)
        get_mol_kegg(kegg_id, self.KEGG_MOL_PATH)

        smiles = None
        compound_mol_file_path = self.KEGG_MOL_PATH / f"{kegg_id}.mol"
        if compound_mol_file_path.exists():
            mol = MolFromMolFile(str(compound_mol_file_path))
            if mol:
                smiles = MolToSmiles(mol)
                try:
                    smiles = standardize_smiles(smiles)
                except:
                    pass
                formula = CalcMolFormula(mol) if not formula else formula

        dblinks = extract_value(r"DBLINKS\s+([\s\S]+?)(?=\n[A-Z]|$)", compound)
        dblinks_dict = None
        if dblinks:
            dblinks_dict = {}
            for elem in dblinks:
                key, value = elem.split(":", 1)
                value = value.split("\n///")[0].strip()
                dblinks_dict[key.strip()] = value

        return {
            "kegg_id": kegg_id,
            "name": name,
            "formula": formula[0] if formula else None,
            "smiles": smiles,
            "dblinks": dblinks_dict,
        }

    def parse_kegg_glycan(self, glycan):
        glycan_id = re.split(r"ENTRY\s+(\S+)", glycan)[1]
        try:
            compound_id = re.split(r"REMARK\s+Same as:\s+(\S+)", glycan)[1]
        except:
            compound_id = None
        return {"kegg_id": glycan_id, "compound_id": compound_id}

    def download_kegg_json(self, ids, path, _type):
        if _type == "reaction":
            parse_fn = self.parse_kegg_reaction
        elif _type == "compound":
            parse_fn = self.parse_kegg_compound
        elif _type == "glycan":
            parse_fn = self.parse_kegg_glycan

        try:
            elems = (
                requests.get(f"https://rest.kegg.jp/get/{'+'.join(ids)}")
                .text.strip()
                .split("///\n")
            )
        except Exception as e:
            self.logger.error(f"Failed to download KEGG {_type}s: {e}")
            raise

        for elem in elems:
            if elem:
                elem_dict = parse_fn(elem)
                save_json(elem_dict, path / f"{elem_dict['kegg_id']}.json")

    def download_kegg_data(self, ids, path, _type, verbose=True):
        download_ids = list(set(ids) - {m.stem for m in path.glob("*.json")})
        if download_ids:
            download_ids = [
                download_ids[i : i + 10] for i in range(0, len(download_ids), 10)
            ]
            for _ in tqdm(
                ThreadPool(2).imap_unordered(
                    lambda ids: self.download_kegg_json(ids, path, _type), download_ids
                ),
                total=len(download_ids),
                desc=f"Downloading KEGG {_type.capitalize()}s",
                disable=not verbose,
            ):
                pass
        return [
            load_json(path / f"{id}.json")
            for id in ids
            if (path / f"{id}.json").exists()
        ]

    def download_kegg_reaction(self, ids, verbose=True):
        return self.download_kegg_data(
            ids, self.KEGG_REACTION_PATH, "reaction", verbose
        )

    def download_kegg_compound(self, ids, verbose=True):
        return self.download_kegg_data(
            ids, self.KEGG_COMPOUND_PATH, "compound", verbose
        )

    def download_kegg_glycan(self, ids, verbose=True):
        return self.download_kegg_data(ids, self.KEGG_GLYCAN_PATH, "glycan", verbose)

    def fill_missing_smiles(self, row, pubchem_data):
        smiles = pubchem_data[row["PubChem"]]["smiles"]
        formula = pubchem_data[row["PubChem"]]["formula"]
        if not smiles and not formula:
            self.logger.warning(
                f"No SMILES or formula found for PubChem ID: {row['PubChem']}"
            )
            return row

        compound = load_json(self.KEGG_COMPOUND_PATH / f"{row['kegg_id']}.json")
        compound["smiles"], compound["formula"] = smiles, formula
        save_json(compound, self.KEGG_COMPOUND_PATH / f"{row['kegg_id']}.json")

        try:
            MolToMolFile(
                MolFromSmiles(smiles), str(self.KEGG_MOL_PATH / f"{row['kegg_id']}.mol")
            )
        except Exception as e:
            self.logger.error(f"Failed to write MOL file for {row['kegg_id']}: {e}")
        row["smiles"], row["formula"] = smiles, formula
        return row

    def generate_equation(self, row, formula_dict):
        def generate_side(side, formula_dict):
            try:
                side_eqn = " + ".join(
                    [
                        f"{elem[0]} {formula_dict[elem[1]]}"
                        for elem in side
                        if elem[1].startswith("C")
                    ]
                )
            except:
                self.logger.error(
                    f"Error generating equation for row {row.name}: {side}"
                )
                side_eqn = None
            return side_eqn

        left_eqn, right_eqn = generate_side(
            row["substrates"], formula_dict
        ), generate_side(row["products"], formula_dict)
        row["equation"] = (
            f"{left_eqn} = {right_eqn}" if left_eqn and right_eqn else None
        )
        return row

    def setup(self):
        self.logger.info("Downloading latest list of KEGG reactions...")
        try:
            kegg_reactions = (
                requests.get(self.KEGG_REACTION_LIST_URL, verify=False)
                .text.strip()
                .split("\n")
            )
        except Exception as e:
            self.logger.error(f"Failed to download KEGG reactions list: {e}")
            raise
        kegg_reactions_df = pd.DataFrame(
            self.download_kegg_reaction([id.split("\t")[0] for id in kegg_reactions])
        )
        self.logger.info("Completed downloading KEGG reactions.")

        kegg_glycans = (
            pd.concat([kegg_reactions_df["substrates"], kegg_reactions_df["products"]])
            .explode()
            .str[-1]
        )
        kegg_glycans = self.download_kegg_glycan(
            kegg_glycans[kegg_glycans.str.startswith("G")].unique().tolist()
        )
        kegg_glycans = {
            glycan["kegg_id"]: glycan["compound_id"]
            for glycan in kegg_glycans
            if glycan["compound_id"]
        }

        if kegg_glycans:
            self.logger.info(
                "Replacing glycan IDs with compound IDs in reaction JSONs..."
            )
            for m in self.KEGG_REACTION_PATH.glob("*.json"):
                original_data, new_data = load_json(m), load_json(m)
                for substrate in new_data["substrates"]:
                    if substrate[-1][0] == "G":
                        try:
                            substrate[-1] = kegg_glycans[substrate[-1]]
                        except KeyError:
                            self.logger.warning(
                                f"Glycan ID {substrate[-1]} not found in glycan mapping."
                            )
                for product in new_data["products"]:
                    if product[-1][0] == "G":
                        try:
                            product[-1] = kegg_glycans[product[-1]]
                        except KeyError:
                            self.logger.warning(
                                f"Glycan ID {product[-1]} not found in glycan mapping."
                            )
                if new_data != original_data:
                    save_json(new_data, m)

            self.logger.info("Reloading updated reaction JSONs into DataFrame...")
            kegg_reactions_df = pd.DataFrame(
                [load_json(m) for m in self.KEGG_REACTION_PATH.glob("*.json")]
            )

        kegg_compounds = (
            pd.concat([kegg_reactions_df["substrates"], kegg_reactions_df["products"]])
            .explode()
            .str[-1]
        )
        kegg_compounds = self.download_kegg_compound(
            kegg_compounds[kegg_compounds.str.startswith("C")].unique().tolist()
        )
        kegg_compounds_df = (
            pd.DataFrame(kegg_compounds).sort_values("kegg_id").reset_index(drop=True)
        )

        missing_smiles = kegg_compounds_df[kegg_compounds_df["smiles"].isna()].copy()
        missing_smiles["PubChem"] = missing_smiles["dblinks"].apply(
            lambda x: x.get("PubChem", None) if isinstance(x, dict) else None
        )
        missing_smiles.dropna(subset=["PubChem"], inplace=True)
        missing_smiles_pubchem_ids = missing_smiles["PubChem"].tolist()
        self.logger.info(
            f"Fetching PubChem data for {len(missing_smiles_pubchem_ids)} compounds missing SMILES..."
        )
        pubchem_data = get_pubchem_compound(missing_smiles_pubchem_ids)
        missing_smiles = missing_smiles.apply(
            lambda row: self.fill_missing_smiles(row, pubchem_data), axis=1
        )
        kegg_compounds_df.update(missing_smiles)
        self.logger.info(f"Number of metabolites: {len(kegg_compounds_df)}")
        self.logger.info(
            f"Number of metabolites without SMILES: {kegg_compounds_df['smiles'].isna().sum()}"
        )

        smiles_dict = {
            row["kegg_id"]: row["smiles"] for _, row in kegg_compounds_df.iterrows()
        }
        kegg_reactions_df["rxn_smiles"] = kegg_reactions_df[
            ["substrates", "products"]
        ].apply(lambda row: generate_reaction_SMILES(row, smiles_dict), axis=1)
        self.logger.info(f"Number of unique reactions: {len(kegg_reactions_df)}")
        self.logger.info(
            f"Number of unique reactions without rxn SMILES: {kegg_reactions_df['rxn_smiles'].isna().sum()}"
        )

        kegg_reactions_df = kegg_reactions_df.explode("ec")
        self.logger.info(f"Number of reactions: {len(kegg_reactions_df)}")
        self.logger.info(
            f"Number of reactions without rxn SMILES: {kegg_reactions_df['rxn_smiles'].isna().sum()}"
        )
        kegg_reactions_df = (
            kegg_reactions_df.explode("rhea_id")
            .sort_values("kegg_id")
            .reset_index(drop=True)
        )

        kegg_reactions_df = (
            kegg_reactions_df.apply(
                lambda row: self.generate_equation(
                    row=row,
                    formula_dict=dict(
                        zip(kegg_compounds_df["kegg_id"], kegg_compounds_df["formula"])
                    ),
                ),
                axis=1,
            )
            .drop(columns=["substrates", "products", "dois"])
            .rename(columns={"pmids": "pubmed_id"})
        )

        try:
            kegg_compounds_df.to_parquet(
                self.KEGG_COMPOUNDS_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info("Successfully saved kegg_metabolites.parquet")
        except Exception as e:
            self.logger.error(f"Failed to save kegg_metabolites.parquet: {e}")
        try:
            kegg_reactions_df.to_parquet(
                self.KEGG_REACTIONS_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info("Successfully saved kegg_reactions.parquet")
        except Exception as e:
            self.logger.error(f"Failed to save kegg_reactions.parquet: {e}")

        return kegg_reactions_df, kegg_compounds_df


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../../configs/"):
        cfg = compose(config_name="data_processing")
    builder = KEGGDatasetBuilder(cfg)
    kegg_reactions_df, kegg_compounds_df = builder.setup()
