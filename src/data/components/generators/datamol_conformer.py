import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from multiprocessing import Pool
from src.utils.tqdmlogger import TqdmLogger


class Datamol3DConformerGenerator:

    def __init__(self, cfg: DictConfig):
        self.RANDOM_SEED = cfg.embeddings.datamol_random_seed
        self.MAX_ATTEMPTS = cfg.embeddings.datamol_max_attempts
        self.NUM_CONFORMERS = cfg.embeddings.datamol_num_conformers
        self.OPTIMIZE = cfg.embeddings.datamol_conformer_optimization

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.datamol_3dconformers_log_file_name,
        ).get_logger()

        self.SDF_PATH = Path(cfg.embeddings.sdf_path)
        self.DATAMOL_SDF_PATH = self.SDF_PATH / "datamol"
        self.MOL_2D_PATH = Path(cfg.embeddings.mol_2d_path)
        self.MOL_3D_PATH = Path(cfg.embeddings.mol_3d_path)

        for path in [LOG_PATH, self.SDF_PATH, self.DATAMOL_SDF_PATH, self.MOL_3D_PATH]:
            path.mkdir(parents=True, exist_ok=True)

    def _datamol_sdf_path(self, smi_hash: str, conf_id: int) -> Path:
        shard_dir = self.DATAMOL_SDF_PATH / smi_hash[:4]
        return shard_dir / f"datamol_{smi_hash}_conf{conf_id}.sdf"

    def _iter_existing_datamol_sdf_files(self):
        if self.DATAMOL_SDF_PATH.exists():
            yield from self.DATAMOL_SDF_PATH.glob("*/*.sdf")

    def _generate_conformer_datamol(self, args):
        import datamol as dm
        from rdkit import Chem

        dm.disable_rdkit_log()

        successful_conformers = 0

        smi_hash, smiles = args
        try:
            mol = dm.to_mol(smiles)
            mol = dm.conformers.generate(
                mol,
                n_confs=self.NUM_CONFORMERS,
                random_seed=self.RANDOM_SEED,
                minimize_energy=self.OPTIMIZE,
                align_conformers=True,
                sort_by_energy=True,
                num_threads=os.cpu_count(),
            )
            conformers = mol.GetConformers()
        except Exception as e:
            self.logger.error(f"Failed to generate conformers for {smiles}: {e}")
            return (smi_hash, False)

        if len(conformers) == 0:
            self.logger.error(f"No valid conformers generated for {smiles}")
            return (smi_hash, False)

        for conf_id in range(len(conformers)):
            try:
                with Chem.SDWriter(
                    str(self._datamol_sdf_path(smi_hash, conf_id))
                ) as writer:
                    writer.write(mol, confId=conf_id)
                successful_conformers += 1
            except Exception as e:
                self.logger.error(f"Failed to write SDF for conformer {conf_id}: {e}")
                continue

        return (smi_hash, successful_conformers > 0)

    def generate(self, smiles_dict):
        """
        Generate 3D conformers for SMILES strings using multiprocessing.

        Args:
            smiles_dict (dict): Dictionary mapping smi_hash -> smiles

        Returns:
            tuple: (successful_smiles, failed_smiles)
        """
        import re
        import pandas as pd

        self.logger.info(
            f"Generating 3D conformers for {len(smiles_dict)} unique SMILES..."
        )

        smiles_df = (
            pd.DataFrame(smiles_dict.items(), columns=["smiles_hash", "smiles"])
            .sort_values("smiles", key=lambda x: x.str.len())
            .reset_index(drop=True)
        )
        smiles_dict = dict(zip(smiles_df["smiles_hash"], smiles_df["smiles"]))
        del smiles_df

        # Track how many datamol conformers already exist per smiles_hash.
        existing_counts: dict[str, set[int]] = {}
        for f in self._iter_existing_datamol_sdf_files():
            m = re.search(r"datamol_([a-f0-9]{64})_conf(\d+)$", f.stem)
            if m:
                h = m.group(1)
                conf_id = int(m.group(2))
                existing_counts.setdefault(h, set()).add(conf_id)

        existing_smi_hashes = {
            h
            for h, conf_ids in existing_counts.items()
            if len(conf_ids) >= self.NUM_CONFORMERS
        }
        self.logger.info(
            f"Found complete existing conformer sets for {len(existing_smi_hashes)} SMILES"
        )

        # Filter out hashes that already have SDFs
        smiles_dict = {
            k: v for k, v in smiles_dict.items() if k not in existing_smi_hashes
        }

        if len(smiles_dict) == 0:
            self.logger.info("All SMILES already have complete 3D conformer sets. Exiting.")
            return

        self.logger.info(
            f"Processing {len(smiles_dict)} new SMILES "
            f"(skipped {len(existing_smi_hashes)} with existing SDFs)"
        )

        failed_smiles = []
        successful_smiles = []

        with tqdm(
            total=len(smiles_dict), desc="Generating 3D conformers", leave=True
        ) as pbar:
            with Pool(processes=os.cpu_count()) as pool:
                for smi_hash, success in pool.imap_unordered(
                    self._generate_conformer_datamol,
                    smiles_dict.items(),
                ):
                    if success:
                        successful_smiles.append(smi_hash)
                    else:
                        failed_smiles.append(smi_hash)
                    pbar.update()

        # Log statistics
        self.logger.info(
            f"Successfully generated conformers for {len(successful_smiles)} SMILES"
        )
        if failed_smiles:
            self.logger.warning(
                f"Failed to generate conformers for {len(failed_smiles)} SMILES"
            )

        return successful_smiles, failed_smiles
