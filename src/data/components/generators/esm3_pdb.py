import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import sys
import torch
import pickle
import contextlib
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
import torch.multiprocessing as mp
from src.utils.tqdmlogger import TqdmLogger


@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr output.
    Useful for suppressing verbose library outputs.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class ESM3PDBGenerator:
    def __init__(self, cfg: DictConfig):
        self.gpu_ids = cfg.embeddings.gpu_ids

        self.ALPHAFOLD_PDB_PATH = Path(cfg.embeddings.af_pdb_path)
        self.ESM3_PDB_PATH = Path(cfg.embeddings.esm_pdb_path)
        self.INPUT_FILE_PATH = cfg.embeddings.esm3_pdbs_input_pkl_path

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.embeddings.esm3_pdbs_log_file_name
        ).get_logger()

        for path in [LOG_PATH, self.ESM3_PDB_PATH]:
            path.mkdir(parents=True, exist_ok=True)

        if not Path(self.INPUT_FILE_PATH).exists():
            self.logger.error(f"Input file not found: {self.INPUT_FILE_PATH}")
            raise FileNotFoundError(f"Input file not found: {self.INPUT_FILE_PATH}")

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        self.logger.info(f"Using GPUs: {self.gpu_ids}")

    def predict_structures(self, gpu_id, proteins, use_cpu=False):
        """
        Generate PDB structures for a list of proteins using ESM3.

        Args:
            gpu_id (int): GPU ID to use for computation.
            proteins (list): List of protein dictionaries with 'sequence' and 'acc_id'.
            use_cpu (bool): Whether to use CPU instead of GPU. Defaults to False.
        """
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, GenerationConfig
        from esm.utils.constants.models import ESM3_OPEN_SMALL

        device = "cpu" if use_cpu else f"cuda:{gpu_id}"  # Determine the device to use
        if not use_cpu:
            try:
                # Set the specified GPU as the current device
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()  # Clear the GPU memory cache
            except Exception as e:
                self.logger.error(
                    f"Failed to set GPU {gpu_id} as the current device: {e}"
                )
                raise e

        # Load the ESM3 model on the specified device
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(device)

        # Iterate over the list of proteins and generate PDB structures
        for protein in tqdm(
            proteins,
            position=gpu_id,
            desc=f"{f'CPU {gpu_id}' if use_cpu else f'GPU {gpu_id}'} - Generating PDBs",
            leave=False,
        ):
            esm_protein = ESMProtein(protein["sequence"])  # Create an ESMProtein object
            structure = None
            try:
                with suppress_output():
                    if not use_cpu:
                        torch.cuda.empty_cache()
                    # Generate the protein structure using the ESM3 model
                    structure = client.generate(
                        esm_protein, GenerationConfig(track="structure", num_steps=32)
                    )
                # Save the generated structure as a PDB file
                structure.to_pdb(
                    self.ESM3_PDB_PATH / f"ESM3-open-small-{protein['acc_id']}.pdb"
                )
            except:
                # Log a warning if structure generation fails
                self.logger.warning(
                    f"Failed to generate structure for {protein['acc_id']} on GPU {gpu_id}."
                )
            finally:
                del structure, esm_protein
                if not use_cpu:
                    torch.cuda.empty_cache()  # Clear the GPU memory cache in case of failure

    def copy_pdbs(self, seq_df, unique_df):
        from shutil import copyfile

        copies = 0
        seq_to_acc_ids = seq_df.groupby("sequence")["acc_id"].apply(list).to_dict()
        for _, row in unique_df.iterrows():
            acc_ids = seq_to_acc_ids[row["sequence"]]
            main_acc_id = row["acc_id"]
            pdb_path = self.ESM3_PDB_PATH / f"ESM3-open-small-{main_acc_id}.pdb"
            for acc_id in acc_ids:
                if acc_id != main_acc_id:
                    target_path = self.ESM3_PDB_PATH / f"ESM3-open-small-{acc_id}.pdb"
                    if target_path.exists():
                        continue
                    try:
                        copyfile(pdb_path, target_path)
                        self.logger.info(f"Copied PDB for {acc_id} from {main_acc_id}.")
                        copies += 1
                    except Exception as e:
                        self.logger.error(
                            f"Failed to copy PDB for {acc_id} from {main_acc_id}: {e}"
                        )
        return copies

    def generate(self):
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Load the protein sequences from the input pickle file
        with open(self.INPUT_FILE_PATH, "rb") as f:
            sequences = pickle.load(f)
        self.logger.info(f"Loaded {len(sequences)} sequences.")

        # Filter sequences to exclude those with existing PDB files
        existing_af_acc_ids = set()
        for pdb_file in self.ALPHAFOLD_PDB_PATH.glob("AF-*-F1-model_v*.pdb"):
            acc_id = pdb_file.stem.split("-")[1]
            existing_af_acc_ids.add(acc_id)
        self.logger.info(
            f"Found {len(existing_af_acc_ids)} existing AlphaFold PDB files"
        )

        # Now filter sequences using fast set lookup
        sequences = [
            prot
            for prot in tqdm(sequences, desc="Filtering out AFDB existing PDBs")
            if prot["acc_id"] not in existing_af_acc_ids
        ]
        self.logger.info(
            f"Filtered sequences to {len(sequences)} that do not have an AFDB PDB file already."
        )

        sequences_df = pd.DataFrame(sequences).sort_values(by=["acc_id"])
        unique_sequences = sequences_df.drop_duplicates(subset=["sequence"])
        self.logger.info(
            f"Filtered sequences to {len(unique_sequences)} unique sequences."
        )
        unique_sequences = unique_sequences[
            unique_sequences["acc_id"].apply(
                lambda x: not (self.ESM3_PDB_PATH / f"ESM3-open-small-{x}.pdb").exists()
            )
        ]
        self.logger.info(
            f"Filtered sequences to {len(unique_sequences)} that do not have a ESM3 generated PDB file already."
        )

        if len(unique_sequences) == 0:
            self.logger.info("No unique sequences to process. Exiting.")
            return

        # Split the sequences into chunks for parallel processing across GPUs
        self.logger.info("Splitting sequences into chunks for each GPU...")
        proteins_chunks = [
            unique_sequences.to_dict(orient="records")[i::num_gpus]
            for i in range(num_gpus)
        ]

        self.logger.info(
            f"Generating PDBs for {len(unique_sequences)} proteins using {num_gpus} GPUs."
        )
        processes = []
        try:
            # Start a separate process for each GPU
            for gpu_id in range(num_gpus):
                p = mp.Process(
                    target=self.predict_structures,
                    args=(gpu_id, proteins_chunks[gpu_id]),
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            # Handle keyboard interruption and terminate all processes
            self.logger.warning("Interrupted! Terminating processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
            self.logger.warning("All processes terminated.")

        self.logger.info(f"{len(unique_sequences)} unique PDBs generated.")
        self.logger.info("Processing PDB files for duplicate acc_ids...")
        copies = self.copy_pdbs(sequences_df, unique_sequences)
        self.logger.info(
            f"Processed PDB files for duplicate acc_ids. Total copies made: {copies}"
        )
