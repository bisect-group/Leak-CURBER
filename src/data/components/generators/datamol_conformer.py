import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hashlib
import os
import time
from pathlib import Path
from src.utils.rich_progress import progress, write
from omegaconf import DictConfig
from multiprocessing import Pool
from src.utils.rich_progress import component_logger


class Datamol3DConformerGenerator:

    def __init__(self, cfg: DictConfig):
        self.RANDOM_SEED = cfg.embeddings.datamol_random_seed
        self.MAX_ATTEMPTS = cfg.embeddings.datamol_max_attempts
        self.NUM_CONFORMERS = cfg.embeddings.datamol_num_conformers
        self.OPTIMIZE = cfg.embeddings.datamol_conformer_optimization
        self.TIMEOUT_S = int(cfg.embeddings.datamol_conformer_timeout_s)

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = component_logger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.datamol_3dconformers_log_file_name,
        ).get_logger()

        self.SDF_PATH = Path(cfg.embeddings.sdf_path)
        self.DATAMOL_SDF_PATH = self.SDF_PATH / "datamol"
        # Append-only "hash<TAB>status<TAB>settings_id" lines, one per attempt without a full set.
        self.OUTCOMES_PATH = self.DATAMOL_SDF_PATH / "outcomes.tsv"
        self.MOL_2D_PATH = Path(cfg.embeddings.mol_2d_path)
        self.MOL_3D_PATH = Path(cfg.embeddings.mol_3d_path)

        for path in [LOG_PATH, self.SDF_PATH, self.DATAMOL_SDF_PATH, self.MOL_3D_PATH]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def _settings_id(self) -> str:
        settings = (
            self.RANDOM_SEED,
            self.NUM_CONFORMERS,
            self.MAX_ATTEMPTS,
            self.OPTIMIZE,
            self.TIMEOUT_S,
        )
        return hashlib.sha256(repr(settings).encode()).hexdigest()[:12]

    def _read_attempted(self) -> set[str]:
        """Hashes whose last attempt under the current settings did not give a full set."""
        if not self.OUTCOMES_PATH.exists():
            return set()
        attempted = set()
        for line in self.OUTCOMES_PATH.read_text().splitlines():
            fields = line.split("\t")
            # A line cut short by a crash has fewer fields or a wrong settings id.
            if len(fields) == 3 and fields[2] == self._settings_id:
                attempted.add(fields[0])
        return attempted

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
        start = time.monotonic()
        try:
            mol = dm.to_mol(smiles)
            mol = dm.conformers.generate(
                mol,
                n_confs=self.NUM_CONFORMERS,
                random_seed=self.RANDOM_SEED,
                minimize_energy=self.OPTIMIZE,
                align_conformers=True,
                sort_by_energy=True,
                # Each call runs inside a pool of cpu_count processes; one thread per call
                # avoids cpu_count^2 threads.
                num_threads=1,
                # RDKit stops the whole embedding call after this many seconds.
                embed_params={"timeout": self.TIMEOUT_S},
            )
            conformers = mol.GetConformers()
        except Exception as e:
            if time.monotonic() - start >= self.TIMEOUT_S:
                self.logger.error(f"Timed out after {self.TIMEOUT_S}s for {smiles}")
                return (smi_hash, "timeout")
            self.logger.error(f"Failed to generate conformers for {smiles}: {e}")
            return (smi_hash, "failed")

        if len(conformers) == 0:
            self.logger.error(f"No valid conformers generated for {smiles}")
            return (smi_hash, "failed")

        # SDWriter fails on a missing shard folder, so create it first.
        self._datamol_sdf_path(smi_hash, 0).parent.mkdir(parents=True, exist_ok=True)
        for conf_id in range(len(conformers)):
            # Write to <final>.sdf.tmp (not matched by *.sdf), then atomically rename it, so a
            # kill mid-write never leaves a truncated SDF that resume counts as done.
            final = self._datamol_sdf_path(smi_hash, conf_id)
            tmp = final.with_name(final.name + ".tmp")
            try:
                with Chem.SDWriter(str(tmp)) as writer:
                    writer.write(mol, confId=conf_id)
                tmp.replace(final)
                successful_conformers += 1
            except Exception as e:
                tmp.unlink(missing_ok=True)
                self.logger.error(f"Failed to write SDF for conformer {conf_id}: {e}")
                continue

        if successful_conformers == 0:
            return (smi_hash, "failed")
        return (smi_hash, "ok" if successful_conformers >= self.NUM_CONFORMERS else "partial")

    def generate(self, smiles_dict):
        """
        Generate 3D conformers for SMILES strings using multiprocessing.

        Args:
            smiles_dict (dict): Dictionary mapping sha256(smiles_key) -> smiles_key

        Molecules that failed, timed out or gave fewer conformers are recorded in
        `outcomes.tsv` with the settings id and are not retried under the same settings.

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

        attempted = self._read_attempted()
        self.logger.info(f"Found {len(attempted)} SMILES already attempted under these settings")
        skipped = existing_smi_hashes | attempted
        n_total = len(smiles_dict)
        smiles_dict = {k: v for k, v in smiles_dict.items() if k not in skipped}

        if len(smiles_dict) == 0:
            self.logger.info("All SMILES already have complete or attempted 3D conformer sets. Exiting.")
            return

        self.logger.info(
            f"Processing {len(smiles_dict)} new SMILES "
            f"(skipped {n_total - len(smiles_dict)} complete or already attempted)"
        )

        failed_smiles = []
        successful_smiles = []

        with progress(
            total=len(smiles_dict), desc="Generating 3D conformers", leave=True
        ) as pbar:
            with Pool(processes=os.cpu_count()) as pool, open(
                self.OUTCOMES_PATH, "a", buffering=1
            ) as outcomes:
                for smi_hash, status in pool.imap_unordered(
                    self._generate_conformer_datamol,
                    smiles_dict.items(),
                ):
                    if status in ("ok", "partial"):
                        successful_smiles.append(smi_hash)
                    else:
                        failed_smiles.append(smi_hash)
                    if status != "ok":
                        outcomes.write(f"{smi_hash}\t{status}\t{self._settings_id}\n")
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
