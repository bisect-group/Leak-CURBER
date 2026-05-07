import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import sys
import torch
import pickle
import warnings
import contextlib
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool
from omegaconf import DictConfig
import torch.multiprocessing as mp
from src.utils.tqdmlogger import TqdmLogger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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


class ESMCEmbedder:
    def __init__(self, cfg: DictConfig):
        self.gpu_ids = cfg.embeddings.gpu_ids

        self.INPUT_FILE_PATH = Path(cfg.embeddings.esmc_embeddings_input_pkl_path)
        self.OUTPUT_FILE_PATH = Path(cfg.embeddings.esmc_embeddings_output_pkl_path)

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.embeddings.esmc_embeddings_log_file_name
        ).get_logger()

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file not found: {self.INPUT_FILE_PATH}")
            raise FileNotFoundError(f"Input file not found: {self.INPUT_FILE_PATH}")

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        self.logger.info(f"Using GPUs: {self.gpu_ids}")

    def get_embeddings(self, sequences, gpu_id):
        """
        Generate embeddings for a list of protein sequences using the ESMC model.

        Args:
            sequences (list): List of protein sequences.
            gpu_id (int): GPU ID to use for computation.

        Returns:
            dict: A dictionary mapping sequences to their embeddings.
        """
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        # Set the GPU device for PyTorch
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()  # Clear the GPU cache

        # Load the pre-trained ESMC model onto the specified GPU
        client = ESMC.from_pretrained("esmc_600m").to(f"cuda:{gpu_id}")

        # Dictionary to store embeddings
        protein_embeddings = {}

        # Process each sequence
        for seq in tqdm(
            sequences,
            position=gpu_id,
            desc=f"GPU {gpu_id} - Processing embeddings",
            leave=False,
        ):
            protein = protein_tensor = logits_output = None
            try:
                # Create an ESMProtein object from the sequence
                protein = ESMProtein(seq)

                # Encode the protein sequence into a tensor
                protein_tensor = client.encode(protein)

                # Generate logits and extract embeddings
                logits_output = (
                    client.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True),
                    )
                    .embeddings.mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                protein_embeddings[seq] = logits_output
            except Exception as e:
                del protein, protein_tensor, logits_output
                torch.cuda.empty_cache()  # Clear GPU cache in case of failure
                self.logger.error(f"Error processing sequence '{seq}': {e}")

        return protein_embeddings

    def process_embeddings(self, gpu_id, sequences_chunk, return_dict):
        """
        Process a chunk of sequences on a specific GPU and store the results in a shared dictionary.

        Args:
            gpu_id (int): GPU ID to use for computation.
            sequences_chunk (list): Chunk of sequences to process.
            return_dict (multiprocessing.Manager().dict): Shared dictionary to store results.
        """
        # Generate embeddings for the chunk of sequences
        embeddings = self.get_embeddings(sequences_chunk, gpu_id)

        # Store the embeddings in the shared dictionary
        return_dict[gpu_id] = embeddings

    def embed(self):
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Number of GPUs available: {num_gpus}")

        # Load the protein sequences from the input file
        self.logger.info(f"Loading sequences from {self.INPUT_FILE_PATH}")
        with open(self.INPUT_FILE_PATH, "rb") as f:
            sequences = pickle.load(f)
            # If it's a list of dicts, extract the 'sequence' value from each dict
            if isinstance(sequences, list):
                if len(sequences) > 0 and isinstance(sequences[0], dict):
                    self.logger.info(
                        "Detected list of dictionaries, extracting 'sequence' values"
                    )
                    try:
                        sequences = [d["sequence"] for d in sequences]
                    except Exception as e:
                        self.logger.error(
                            f"Error extracting 'sequence' from dictionaries: {e}"
                        )
                        raise ValueError(
                            "Input sequences must be a list of strings or a list of dictionaries with 'sequence' keys."
                        )

                elif len(sequences) > 0 and not isinstance(sequences[0], str):
                    self.logger.error(
                        f"Unsupported sequence format. Expected list of strings or list of dicts with 'sequence' keys., got list of {type(sequences[0])}"
                    )
                    raise ValueError(
                        f"Unsupported sequence format: Expected list of strings or list of dicts with 'sequence' keys., got {type(sequences[0])}"
                    )

            elif not isinstance(sequences, list):
                self.logger.error(
                    f"Unsupported sequence file format: {type(sequences)}. Expected list of strings or list of dicts with 'sequence' keys.."
                )
                raise ValueError(
                    f"Unsupported sequence file format: {type(sequences)} . Expected list of strings or list of dicts with 'sequence' keys.."
                )
        self.logger.info(
            f"Loaded {len(sequences)} sequences from {self.INPUT_FILE_PATH}"
        )
        existing_embeddings, existing_sequences = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.info(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_sequences = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_sequences)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        sequences = list(set(sequences) - existing_sequences)
        self.logger.info(
            f"{len(sequences)} sequences remaining to compute excluding existing embeddings"
        )

        if len(sequences) == 0:
            self.logger.info("All sequence embeddings already exist. Exiting.")
            return

        # Split the sequences into chunks for parallel processing
        self.logger.info(
            f"Splitting sequences into {num_gpus} chunks for multiprocessing"
        )
        sequences_chunks = [sequences[i::num_gpus] for i in range(num_gpus)]

        # Create a multiprocessing manager and shared dictionary
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        try:
            # Start a separate process for each GPU
            for gpu_id in range(num_gpus):
                p = mp.Process(
                    target=self.process_embeddings,
                    args=(gpu_id, sequences_chunks[gpu_id], return_dict),
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Combine the results from all GPUs
            protein_embeddings = {}
            for gpu_id in range(num_gpus):
                protein_embeddings.update(return_dict[gpu_id])

            protein_embeddings.update(existing_embeddings)
            # Save the embeddings to a pkl file
            with open(self.OUTPUT_FILE_PATH, "wb") as f:
                pickle.dump(protein_embeddings, f)
            self.logger.info(f"Sequence embeddings saved to {self.OUTPUT_FILE_PATH}")

        except KeyboardInterrupt:
            # Handle keyboard interrupt and terminate processes
            self.logger.error("Interrupted! Terminating processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
            self.logger.error("All processes terminated.")


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
        self.MOL_2D_PATH = Path(cfg.embeddings.mol_2d_path)
        self.MOL_3D_PATH = Path(cfg.embeddings.mol_3d_path)

        for path in [LOG_PATH, self.SDF_PATH, self.MOL_3D_PATH]:
            path.mkdir(parents=True, exist_ok=True)

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
                    str(self.SDF_PATH / f"datamol_{smi_hash}_conf{conf_id}.sdf")
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

        existing_sdf_files = list(self.SDF_PATH.glob("*.sdf"))
        # Extract sha256 hash (64 hex chars) from any filename format
        existing_smi_hashes = {
            match.group(0)
            for f in existing_sdf_files
            if (match := re.search(r"[a-f0-9]{64}", f.stem))
        }
        self.logger.info(
            f"Found existing SDF files for {len(existing_smi_hashes)} SMILES"
        )

        # Filter out hashes that already have SDFs
        smiles_dict = {
            k: v for k, v in smiles_dict.items() if k not in existing_smi_hashes
        }

        if len(smiles_dict) == 0:
            self.logger.info("All SMILES already have 3D conformers. Exiting.")
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


class ESM3Embedder:
    def __init__(self, cfg: DictConfig):
        self.gpu_ids = cfg.embeddings.gpu_ids

        self.ALPHAFOLD_PDB_PATH = Path(cfg.embeddings.af_pdb_path)
        self.ESM3_PDB_PATH = Path(cfg.embeddings.esm_pdb_path)
        self.PROCESSED_EXP_PDB_PATH = Path(cfg.embeddings.processed_exp_pdb_path)
        self.INPUT_FILE_PATH = Path(cfg.embeddings.esm3_embeddings_input_pkl_path)
        self.OUTPUT_FILE_PATH = Path(cfg.embeddings.esm3_embeddings_output_pkl_path)

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.embeddings.esm3_embeddings_log_file_name
        ).get_logger()

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file not found: {self.INPUT_FILE_PATH}")
            raise FileNotFoundError(f"Input file not found: {self.INPUT_FILE_PATH}")
        if not self.ALPHAFOLD_PDB_PATH.exists():
            self.logger.error(
                f"AlphaFold PDB path not found: {self.ALPHAFOLD_PDB_PATH}"
            )
            raise FileNotFoundError(
                f"AlphaFold PDB path not found: {self.ALPHAFOLD_PDB_PATH}"
            )
        if not self.ESM3_PDB_PATH.exists():
            self.logger.error(f"ESM3 PDB path not found: {self.ESM3_PDB_PATH}")
            raise FileNotFoundError(f"ESM3 PDB path not found: {self.ESM3_PDB_PATH}")
        if not self.PROCESSED_EXP_PDB_PATH.exists():
            self.logger.error(
                f"Processed experimental PDB path not found: {self.PROCESSED_EXP_PDB_PATH}"
            )
            raise FileNotFoundError(
                f"Processed experimental PDB path not found: {self.PROCESSED_EXP_PDB_PATH}"
            )

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        self.logger.info(f"Using GPUs: {self.gpu_ids}")

    def get_embeddings(self, acc_ids, gpu_id):
        """
        Generate embeddings for a list of protein accession IDs using ESM3 model.

        Args:
            acc_ids (list): List of protein accession IDs.
            gpu_id (int): GPU ID to use for processing.

        Returns:
            dict: Dictionary mapping accession IDs to their embeddings.
        """
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, SamplingConfig
        from esm.utils.constants.models import ESM3_OPEN_SMALL

        # Set the GPU device and clear CUDA cache
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

        # Load the ESM3 model on the specified GPU
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(f"cuda:{gpu_id}")
        protein_embeddings = {}

        # Iterate over each accession ID
        for acc in tqdm(
            acc_ids, position=gpu_id, desc=f"GPU {gpu_id} - Processing embeddings"
        ):
            # Define paths for AlphaFold and ESM PDB files
            af_pdb_files = list(
                self.ALPHAFOLD_PDB_PATH.glob(f"AF-{acc}-F1-model_v*.pdb")
            )
            af_pdb_path = (
                sorted(af_pdb_files, key=lambda x: int(x.stem.split("_v")[-1]))[-1]
                # Sort by version number and take the highest version
                if af_pdb_files
                else None
            )
            esm_pdb_path = self.ESM3_PDB_PATH / f"ESM3-open-small-{acc}.pdb"

            processed_exp_pdb_path = self.PROCESSED_EXP_PDB_PATH / f"{acc}.pdb"

            # Use AlphaFold PDB if it exists, otherwise use ESM PDB
            pdb_path = (
                processed_exp_pdb_path
                if processed_exp_pdb_path.exists()
                else (af_pdb_path if af_pdb_path else esm_pdb_path)
            )

            # Skip if no PDB file is found
            if not pdb_path.exists():
                self.logger.warning(
                    f"Skipping {acc}: PDB file not found in processed experimental, AlphaFold, or ESM3 directories."
                )
                continue

            protein = protein_tensor = output = None
            try:
                # Load the protein structure from the PDB file
                protein = ESMProtein.from_pdb(pdb_path)

                # Encode the protein structure into a tensor
                protein_tensor = client.encode(protein)

                # Generate embeddings using the model
                output = client.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(
                        return_per_residue_embeddings=False, return_mean_embedding=True
                    ),
                )

                # Store the mean embedding in the dictionary
                protein_embeddings[acc] = output.mean_embedding.cpu().numpy()
            except Exception as e:
                # Log any errors encountered during processing
                self.logger.error(f"Error processing {acc} on GPU {gpu_id}: {e}")
                pass
            finally:
                del protein, protein_tensor, output
                torch.cuda.empty_cache()  # Clear the GPU memory cache

        return protein_embeddings

    def process_embeddings(self, gpu_id, acc_ids_chunk, return_dict):
        """
        Process a chunk of accession IDs on a specific GPU and store results in a shared dictionary.

        Args:
            gpu_id (int): GPU ID to use for processing.
            acc_ids_chunk (list): Chunk of accession IDs to process.
            return_dict (multiprocessing.Manager().dict): Shared dictionary to store results.
        """
        # Generate embeddings for the chunk of accession IDs
        embeddings = self.get_embeddings(acc_ids_chunk, gpu_id)

        # Store the embeddings in the shared dictionary
        return_dict[gpu_id] = embeddings

    def embed(self):
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Number of GPUs available: {num_gpus}")

        # Load protein accession IDs from the input file
        self.logger.info(f"Loading protein IDs from {self.INPUT_FILE_PATH}")
        with open(self.INPUT_FILE_PATH, "rb") as f:
            acc_ids = pickle.load(f)

            if isinstance(acc_ids, list):
                if len(acc_ids) > 0 and isinstance(acc_ids[0], dict):
                    self.logger.info(
                        "Detected list of dictionaries, extracting 'acc_id' values"
                    )
                    try:
                        acc_ids = [d["acc_id"] for d in acc_ids]
                    except Exception as e:
                        self.logger.error(
                            f"Error extracting 'acc_id' from dictionaries: {e}"
                        )
                        raise ValueError(
                            "Input sequences must be a list of strings or a list of dictionaries with 'acc_id' keys."
                        )

                elif len(acc_ids) > 0 and not isinstance(acc_ids[0], str):
                    self.logger.error(
                        f"Unsupported sequence format. Expected list of strings or list of dicts with 'acc_id' keys., got list of {type(acc_ids[0])}"
                    )
                    raise ValueError(
                        f"Unsupported sequence format: Expected list of strings or list of dicts with 'acc_id' keys., got {type(acc_ids[0])}"
                    )

            elif not isinstance(acc_ids, list):
                self.logger.error(
                    f"Unsupported sequence file format: {type(acc_ids)}. Expected list of strings or list of dicts with 'acc_id' keys."
                )
                raise ValueError(
                    f"Unsupported sequence file format: {type(acc_ids)}. Expected list of strings or list of dicts with 'acc_id' keys."
                )
        self.logger.info(f"Loaded {len(acc_ids)} protein IDs")

        existing_embeddings = {}
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.info(
                f"Existing output file found: {self.OUTPUT_FILE_PATH}, loading..."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
            self.logger.info(f"Loaded {len(existing_embeddings)} existing embeddings.")

        acc_ids_to_process = list(set(acc_ids) - set(existing_embeddings.keys()))
        if len(acc_ids_to_process) == 0:
            self.logger.info("All protein IDs already have embeddings. Exiting.")
            return
        else:
            self.logger.info(
                f"{len(acc_ids_to_process)} protein IDs need new embeddings."
            )

        # Split the accession IDs into chunks for multiprocessing
        self.logger.info(
            f"Splitting protein IDs into {num_gpus} chunks for multiprocessing"
        )
        acc_ids_chunks = [acc_ids_to_process[i::num_gpus] for i in range(num_gpus)]

        # Create a multiprocessing manager and shared dictionary
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        try:
            # Start a process for each GPU
            for gpu_id in range(num_gpus):
                p = mp.Process(
                    target=self.process_embeddings,
                    args=(gpu_id, acc_ids_chunks[gpu_id], return_dict),
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Combine results from all GPUs
            protein_embeddings = {}
            for gpu_id in range(num_gpus):
                protein_embeddings.update(return_dict.get(gpu_id, {}))

            protein_embeddings.update(existing_embeddings)

            with open(self.OUTPUT_FILE_PATH, "wb") as f:
                pickle.dump(protein_embeddings, f)
            self.logger.info(f"Structure embeddings saved to {self.OUTPUT_FILE_PATH}")

        except KeyboardInterrupt:
            # Handle keyboard interrupt and terminate processes
            self.logger.error("Interrupted! Terminating processes...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
            self.logger.error("All processes terminated.")


class BARTSmilesEmbedder:
    def __init__(self, cfg: DictConfig):
        from transformers import AutoTokenizer, AutoModel, pipeline

        self.BATCH_SIZE = cfg.embeddings.bartsmiles_embeddings_batch_size

        self.INPUT_FILE_PATH = Path(cfg.embeddings.bartsmiles_embeddings_input_pkl_path)
        self.OUTPUT_FILE_PATH = Path(
            cfg.embeddings.bartsmiles_embeddings_output_pkl_path
        )

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.bartsmiles_embeddings_log_file_name,
        ).get_logger()

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file not found: {self.INPUT_FILE_PATH}")
            raise FileNotFoundError(f"Input file not found: {self.INPUT_FILE_PATH}")

        # Load the BARTSmiles model and tokenizer
        self.logger.info("Loading BARTSmiles model...")
        try:
            # Load the tokenizer for the BARTSmiles model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gayane/BARTSmiles", add_prefix_space=True
            )
            self.tokenizer.pad_token = "<pad>"  # Set the padding token explicitly

            # Load the BARTSmiles model
            self.model = AutoModel.from_pretrained("gayane/BARTSmiles")
            self.model.eval()  # Set the model to evaluation mode

            # Create a pipeline for feature extraction
            self.extractor = pipeline(
                "feature-extraction",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as e:
            self.logger.error(f"Error while loading BARTSmiles model: {e}")
            raise e
        self.logger.info(
            f"BARTSmiles model loaded successfully on {self.extractor.device} device."
        )

    def _tokens_less_than_128(self, smi):
        """
        Check if the number of tokens in the SMILES string is less than 128.

        Args:
            smi (str): A SMILES string.

        Returns:
            bool: True if the number of tokens is less than or equal to 128, False otherwise.
        """
        # Tokenize the SMILES string and check the number of tokens
        return (
            self.tokenizer(
                smi,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"].shape[1]
            <= 128
        )

    def _get_bartsmiles_embedding(
        self,
        smiles,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Get the BARTSmiles embedding for a list of SMILES strings using AutoModel directly.

        Args:
            smiles (list): List of SMILES strings.
            batch_size (int): Batch size for processing SMILES strings.
            device (str): Device to use for computation ('cuda' or 'cpu').

        Returns:
            dict: A dictionary mapping SMILES strings to their embeddings.
        """
        # Move the model to the specified device
        self.model.to(device)

        # Filter out SMILES strings with token lengths > 128
        valid_smiles = [
            smiles[i]
            for i in tqdm(
                range(len(smiles)), desc="Filtering SMILES strings with >128 tokens"
            )
            if self._tokens_less_than_128(smiles[i])
        ]
        if len(valid_smiles) == 0:
            self.logger.error("No valid SMILES strings with <= 128 tokens found.")
            return {}
        self.logger.info(
            f"Filtered out {len(smiles) - len(valid_smiles)} SMILES strings with token lengths > 128."
        )
        self.logger.info(f"Number of valid SMILES strings: {len(valid_smiles)}")
        smiles = list(set(valid_smiles))
        self.logger.info(f"Number of unique SMILES strings: {len(smiles)}")

        # Tokenize the SMILES strings
        self.logger.info("Tokenizing SMILES strings...")
        tokenized = self.tokenizer(
            smiles,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if "token_type_ids" in tokenized:
            del tokenized[
                "token_type_ids"
            ]  # Remove token_type_ids if present (not needed for BART)

        # Move tokenized inputs to the specified device
        tokenized = {key: val.to(device) for key, val in tokenized.items()}

        # Initialize variables for storing embeddings and tracking failures
        smiles_embedding = {}
        failed_count = 0

        # Process valid SMILES strings in batches
        self.logger.info(
            f"Getting BARTSmiles embedding for {len(smiles)} SMILES strings..."
        )
        for i in tqdm(
            range(0, len(smiles), self.BATCH_SIZE), desc="Embedding SMILES batches"
        ):
            batch_smiles = smiles[i : i + self.BATCH_SIZE]
            try:
                # Extract embeddings directly from the model
                with torch.no_grad():
                    outputs = self.model(
                        **{
                            key: val[i : i + self.BATCH_SIZE]
                            for key, val in tokenized.items()
                        }
                    )
                    batch_embeddings = outputs.last_hidden_state.mean(
                        dim=1
                    )  # Mean pooling over sequence length

                # Store embeddings in the dictionary
                for j, smile in enumerate(batch_smiles):
                    smiles_embedding[smile] = batch_embeddings[j].cpu().numpy()
            except Exception as e:
                failed_count += len(batch_smiles)
                self.logger.error(
                    f"Error while processing batch {i // self.BATCH_SIZE + 1}: {e}"
                )

        self.logger.info(
            f"Successfully embedded {len(smiles_embedding)} SMILES strings."
        )
        if failed_count > 0:
            self.logger.info(f"Failed to embed {failed_count} SMILES strings.")
        return smiles_embedding

    def embed(self):
        """
        Embed the SMILES strings using BARTSmiles and save the embeddings to a file.

        This function reads SMILES strings from a JSON file, generates embeddings using the BARTSmiles model,
        and saves the embeddings to a pickle file.
        """
        with open(self.INPUT_FILE_PATH, "rb") as f:
            smiles = pickle.load(f)

        # If it's a list of dicts, extract the 'smiles' value from each dict
        if isinstance(smiles, list):
            if len(smiles) > 0 and isinstance(smiles[0], dict):
                self.logger.info(
                    "Detected list of dictionaries, extracting 'smiles' values"
                )
                try:
                    smiles = [d["smiles"] for d in smiles]
                except Exception as e:
                    self.logger.error(
                        f"Error extracting 'smiles' from dictionaries: {e}"
                    )
                    raise ValueError(
                        "Input sequences must be a list of strings or a list of dictionaries with 'smiles' keys."
                    )

            elif len(smiles) > 0 and not isinstance(smiles[0], str):
                self.logger.error(
                    f"Unsupported smiles format. Expected list of strings or list of dicts with 'smiles' keys., got list of {type(smiles[0])}"
                )
                raise ValueError(
                    f"Unsupported smiles format: Expected list of strings or list of dicts with 'smiles' keys., got {type(smiles[0])}"
                )

        elif not isinstance(smiles, list):
            self.logger.error(
                f"Unsupported smiles file format: {type(smiles)}. Expected list of strings or list of dicts with 'smiles' keys.."
            )
            raise ValueError(
                f"Unsupported smiles file format: {type(smiles)} . Expected list of strings or list of dicts with 'smiles' keys.."
            )
        self.logger.info(f"Loaded {len(smiles)} smiles from {self.INPUT_FILE_PATH}")

        existing_embeddings, existing_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.info(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_smiles)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        smiles = list(set(smiles) - existing_smiles)
        self.logger.info(
            f"{len(smiles)} smiles remaining to compute excluding existing embeddings"
        )
        if len(smiles) == 0:
            self.logger.info("All SMILES embeddings already exist. Exiting.")
            return

        # Generate embeddings for the SMILES strings
        smiles_embedding = self._get_bartsmiles_embedding(smiles)
        self.logger.info("SMILES strings embedded successfully.")

        smiles_embedding.update(existing_embeddings)
        try:
            with open(self.OUTPUT_FILE_PATH, "wb") as f:
                pickle.dump(smiles_embedding, f)
            self.logger.info(f"SMILES embeddings saved to {self.OUTPUT_FILE_PATH}")
        except Exception as e:
            self.logger.error(f"Error while saving SMILES embeddings: {e}")
            raise e

        self.logger.info("SMILES embedding process completed successfully.")


class RxnFPEmbedder:
    def __init__(self, cfg: DictConfig):
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator,
            get_default_model_and_tokenizer,
        )

        self.BATCH_SIZE = cfg.embeddings.rxnfp_batch_size

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.INPUT_FILE_PATH = Path(cfg.embeddings.rxnfp_embeddings_input_pkl_path)
        self.OUTPUT_FILE_PATH = Path(cfg.embeddings.rxnfp_embeddings_output_pkl_path)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.rxnfp_embeddings_log_file_name,
        ).get_logger()

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file not found: {self.INPUT_FILE_PATH}")
            raise FileNotFoundError(f"Input file not found: {self.INPUT_FILE_PATH}")

        self.logger.info("Loading RXNFP model...")
        try:
            # Load the default RXNFP model and tokenizer
            model, tokenizer = get_default_model_and_tokenizer()
            self.rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
        except Exception as e:
            self.logger.error(f"Error while loading RXNFP model: {e}")
            raise e

    def get_rxnfp_embeddings(self, rxn_smiles):
        rxn_fps = []
        for i in tqdm(range(0, len(rxn_smiles), self.BATCH_SIZE)):
            batch = rxn_smiles[i : i + self.BATCH_SIZE]
            try:
                batch_fps = self.rxnfp_generator.convert_batch(batch)
                rxn_fps.extend(batch_fps)

            except Exception as e:
                print(f"Error processing batch {i//self.BATCH_SIZE}: {e}")
                continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dict(zip(rxn_smiles, rxn_fps))

    def embed(self):
        """
        Main function to generate RXNFP embeddings for reaction SMILES.

        This function reads reaction SMILES from a specified input file, generates their embeddings
        using the RXNFP model, and saves the embeddings to an output file.
        """

        with open(self.INPUT_FILE_PATH, "rb") as f:
            rxn_smiles = pickle.load(f)

        self.logger.info(
            f"Loaded {len(rxn_smiles)} reaction SMILES from {self.INPUT_FILE_PATH}."
        )

        existing_embeddings, existing_rxn_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.info(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute..."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_rxn_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_rxn_smiles)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        rxn_smiles = list(set(rxn_smiles) - existing_rxn_smiles)
        self.logger.info(
            f"{len(rxn_smiles)} reaction SMILES remaining to compute excluding existing embeddings."
        )

        if len(rxn_smiles) == 0:
            self.logger.info("All reaction SMILES already have embeddings. Exiting.")
            return

        self.logger.info("Generating RXNFP embeddings...")
        reaction_smiles_embedding = self.get_rxnfp_embeddings(rxn_smiles)
        self.logger.info(
            f"Generated embeddings for {len(reaction_smiles_embedding)} reaction SMILES."
        )

        reaction_smiles_embedding.update(existing_embeddings)
        try:
            self.logger.info(f"Saving RXNFP embeddings to {self.OUTPUT_FILE_PATH}...")
            with open(self.OUTPUT_FILE_PATH, "wb") as f:
                pickle.dump(reaction_smiles_embedding, f)
            self.logger.info("RXNFP embeddings saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving RXNFP embeddings: {e}")
            raise e

        self.logger.info("Embedding process completed successfully.")


class MolR2DEmbedder:
    def __init__(self, cfg: DictConfig):
        self.BATCH_SIZE = cfg.embeddings.molr_batch_size

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.INPUT_FILE_PATH = Path(cfg.embeddings.molr_embeddings_input_pkl_path)
        self.OUTPUT_FILE_PATH = Path(cfg.embeddings.molr_embeddings_output_pkl_path)

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.embeddings.molr_embeddings_log_file_name
        ).get_logger()

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file {self.INPUT_FILE_PATH} does not exist.")
            raise FileNotFoundError(
                f"Input file {self.INPUT_FILE_PATH} does not exist."
            )

    @staticmethod
    def molr_worker_init():
        from MolR.featurizer import MolEFeaturizer

        global model
        model = MolEFeaturizer()

    @staticmethod
    def molr_worker(smiles_batch):
        # Use the global model instance
        try:
            embeddings, flags = model.transform(smiles_batch)
            return {
                smile: embedding for smile, embedding in zip(smiles_batch, embeddings)
            }
        except:
            return {}

    def get_molr_2d_embeddings(self, smiles):
        """
        Generate 2D molecular embeddings for a list of SMILES strings using the MolR model.

        This function processes the input SMILES strings in batches, generating 2D embeddings for each molecule
        using the MolEFeaturizer from the MolR package. It logs progress and errors, and returns a dictionary
        mapping each SMILES string to its corresponding embedding.

        Args:
            smiles (list of str): List of SMILES strings representing molecules for which 2D embeddings are to be generated.
            batch_size (int, optional): Number of SMILES strings to process in each batch. Default is 500.

        Returns:
            dict: A dictionary where keys are SMILES strings and values are their corresponding 2D embeddings (numpy arrays or tensors).
        """

        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        # Split SMILES into batches for workers
        chunks = list(chunked(smiles, self.BATCH_SIZE))
        self.logger.info(
            f"Splitting SMILES into {len(chunks)} batches of up to {self.BATCH_SIZE} each for multiprocessing."
        )
        with Pool(
            processes=os.cpu_count() - 1, initializer=self.molr_worker_init
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.molr_worker, chunks),
                    total=len(chunks),
                    desc="2D Embedding Batches",
                )
            )
        # Merge results
        mol_2d_embeddings = {}
        for res in results:
            mol_2d_embeddings.update(res)
        self.logger.info(
            f"Generated 2D embeddings for {len(mol_2d_embeddings)} SMILES strings."
        )
        return mol_2d_embeddings

    def embed(self):
        with open(self.INPUT_FILE_PATH, "rb") as f:
            smiles = pickle.load(f)

        # If it's a list of dicts, extract the 'smiles' value from each dict
        if isinstance(smiles, list):
            if len(smiles) > 0 and isinstance(smiles[0], dict):
                self.logger.info(
                    "Detected list of dictionaries, extracting 'smiles' values"
                )
                try:
                    smiles = [d["smiles"] for d in smiles]
                except Exception as e:
                    self.logger.error(
                        f"Error extracting 'smiles' from dictionaries: {e}"
                    )
                    raise ValueError(
                        "Input sequences must be a list of strings or a list of dictionaries with 'smiles' keys."
                    )

            elif len(smiles) > 0 and not isinstance(smiles[0], str):
                self.logger.error(
                    f"Unsupported smiles format. Expected list of strings or list of dicts with 'smiles' keys., got list of {type(smiles[0])}"
                )
                raise ValueError(
                    f"Unsupported smiles format: Expected list of strings or list of dicts with 'smiles' keys., got {type(smiles[0])}"
                )

        elif not isinstance(smiles, list):
            self.logger.error(
                f"Unsupported smiles file format: {type(smiles)}. Expected list of strings or list of dicts with 'smiles' keys.."
            )
            raise ValueError(
                f"Unsupported smiles file format: {type(smiles)} . Expected list of strings or list of dicts with 'smiles' keys.."
            )
        self.logger.info(f"Loaded {len(smiles)} smiles from {self.INPUT_FILE_PATH}")

        existing_embeddings, existing_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.warning(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_embeddings)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        smiles = list(set(smiles) - existing_smiles)
        if len(smiles) == 0:
            self.logger.info("All SMILES embeddings already exist. Exiting.")
            return
        self.logger.info(f"Computing embeddings for {len(smiles)} new smiles")

        try:
            mol_2d_embeddings = self.get_molr_2d_embeddings(smiles)
        except Exception as e:
            self.logger.error(f"Error during 2D embedding generation: {e}")
            raise e

        mol_2d_embeddings.update(existing_embeddings)
        with open(self.OUTPUT_FILE_PATH, "wb") as f:
            pickle.dump(mol_2d_embeddings, f)
        self.logger.info(
            f"Saved {len(mol_2d_embeddings)} embeddings to {self.OUTPUT_FILE_PATH}"
        )


class ClampSmilesEmbedder:
    def __init__(self, cfg: DictConfig):
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        self.BATCH_SIZE = cfg.embeddings.clamp_batch_size

        self.CLAMP_HP_URL = cfg.embeddings.clamp_hp_url
        self.CLAMP_CHECKPOINT_URL = cfg.embeddings.clamp_checkpoint_url

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.CLAMP_DIR = Path(cfg.embeddings.clamp_dir)

        self.INPUT_FILE_PATH = Path(
            cfg.embeddings.clamp_smiles_embeddings_input_pkl_path
        )
        self.OUTPUT_FILE_PATH = Path(
            cfg.embeddings.clamp_smiles_embeddings_output_pkl_path
        )

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.clamp_smiles_embeddings_log_file_name,
        ).get_logger()

        for path in [LOG_PATH, self.CLAMP_DIR, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file {self.INPUT_FILE_PATH} does not exist.")
            raise FileNotFoundError(
                f"Input file {self.INPUT_FILE_PATH} does not exist."
            )

        self._download_if_needed(self.CLAMP_HP_URL, self.CLAMP_DIR / "hp.json")
        self._download_if_needed(
            self.CLAMP_CHECKPOINT_URL, self.CLAMP_DIR / "checkpoint.pt"
        )

    def _download_if_needed(self, url, dest):
        import subprocess

        if Path(dest).exists():
            return
        self.logger.info(f"Downloading {url} -> {dest}")
        subprocess.run(["curl", "-Lo", str(dest), str(url)], check=True)
        self.logger.info(f"Downloaded: {dest}")

    @staticmethod
    def clamp_worker_init(clamp_dir):
        import clamp
        import logging

        # Reduce CLAMP logger verbosity
        try:
            log = logging.getLogger("clamp")
            log.setLevel(logging.WARNING)
            log.propagate = False
            for h in list(log.handlers):
                log.removeHandler(h)
        except Exception:
            pass

        global model
        model = clamp.CLAMP(device="cpu", path_dir=str(clamp_dir))
        model.eval()

    @staticmethod
    def clamp_worker(smiles_batch):
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings

        try:
            with torch.no_grad():
                compound_features = model.prepro_smiles(smiles_batch)
                batch_emb = model.compound_encoder(compound_features)
            return {
                smi: emb.detach().cpu().numpy()
                for smi, emb in zip(smiles_batch, batch_emb)
            }
        except:
            return {}

    def get_clamp_smiles_embeddings(self, smiles):
        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        chunks = list(chunked(smiles, self.BATCH_SIZE))
        self.logger.info(
            f"Splitting SMILES into {len(chunks)} batches of up to {self.BATCH_SIZE} each for multiprocessing."
        )
        with Pool(
            processes=os.cpu_count() - 1,
            initializer=self.clamp_worker_init,
            initargs=(self.CLAMP_DIR,),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.clamp_worker, chunks),
                    total=len(chunks),
                    desc="CLAMP SMILES Embedding Batches",
                )
            )

        clamp_embeddings = {}
        for res in results:
            clamp_embeddings.update(res)
        self.logger.info(
            f"Generated CLAMP SMILES embeddings for {len(clamp_embeddings)} SMILES strings."
        )
        return clamp_embeddings

    def embed(self):
        with open(self.INPUT_FILE_PATH, "rb") as f:
            smiles = pickle.load(f)

        # If it's a list of dicts, extract the 'smiles' value from each dict
        if isinstance(smiles, list):
            if len(smiles) > 0 and isinstance(smiles[0], dict):
                self.logger.info(
                    "Detected list of dictionaries, extracting 'smiles' values"
                )
                try:
                    smiles = [d["smiles"] for d in smiles]
                except Exception as e:
                    self.logger.error(
                        f"Error extracting 'smiles' from dictionaries: {e}"
                    )
                    raise ValueError(
                        "Input sequences must be a list of strings or a list of dictionaries with 'smiles' keys."
                    )

            elif len(smiles) > 0 and not isinstance(smiles[0], str):
                self.logger.error(
                    f"Unsupported smiles format. Expected list of strings or list of dicts with 'smiles' keys., got list of {type(smiles[0])}"
                )
                raise ValueError(
                    f"Unsupported smiles format: Expected list of strings or list of dicts with 'smiles' keys., got {type(smiles[0])}"
                )

        elif not isinstance(smiles, list):
            self.logger.error(
                f"Unsupported smiles file format: {type(smiles)}. Expected list of strings or list of dicts with 'smiles' keys.."
            )
            raise ValueError(
                f"Unsupported smiles file format: {type(smiles)} . Expected list of strings or list of dicts with 'smiles' keys.."
            )
        self.logger.info(f"Loaded {len(smiles)} smiles from {self.INPUT_FILE_PATH}")

        existing_embeddings, existing_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.warning(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_embeddings)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        smiles = list(set(smiles) - existing_smiles)
        if len(smiles) == 0:
            self.logger.info("All SMILES embeddings already exist. Exiting.")
            return
        self.logger.info(f"Computing embeddings for {len(smiles)} new smiles")

        try:
            clamp_embeddings = self.get_clamp_smiles_embeddings(smiles)
        except Exception as e:
            self.logger.error(f"Error occurred during CLAMP embedding: {e}")
            return

        clamp_embeddings.update(existing_embeddings)
        try:
            with open(self.OUTPUT_FILE_PATH, "wb") as f:
                pickle.dump(clamp_embeddings, f)
            self.logger.info(f"Saved CLAMP embeddings to {self.OUTPUT_FILE_PATH}")
        except Exception as e:
            self.logger.error(f"Error while saving CLAMP embeddings: {e}")
            raise e


class SmilesECFPEmbedder:
    def __init__(self, cfg: DictConfig):
        self.ECFP_RADIUS = cfg.embeddings.ecfp_radius or 2
        self.ECFP_NBITS = cfg.embeddings.ecfp_nbits or 1024

        self.PANDARALLEL_NB_WORKERS = (
            cfg.embeddings.atom_pair_pandarallel.nb_workers
            if cfg.embeddings.atom_pair_pandarallel.nb_workers
            else os.cpu_count()
        )
        self.PANDARALLEL_PROGRESS_BAR = (
            cfg.embeddings.atom_pair_pandarallel.progress_bar
            if cfg.embeddings.atom_pair_pandarallel.progress_bar is not None
            else True
        )

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.INPUT_FILE_PATH = Path(
            cfg.embeddings.smiles_ecfp_embeddings_input_pkl_path
        )
        self.OUTPUT_FILE_PATH = Path(
            cfg.embeddings.smiles_ecfp_embeddings_output_pkl_path
        )

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.smiles_ecfp_embeddings_log_file_name,
        ).get_logger()

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file {self.INPUT_FILE_PATH} does not exist.")
            raise FileNotFoundError(
                f"Input file {self.INPUT_FILE_PATH} does not exist."
            )

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

    def get_smiles_ecfp_embeddings(self, smiles):
        from pandarallel import pandarallel
        from src.utils.chem_utils import ChemUtils

        chem_utils = ChemUtils()
        smiles_to_ecfp = chem_utils.smiles_to_ecfp
        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )

        df = pd.DataFrame(smiles, columns=["smiles"])
        self.logger.info(
            f"Generating ECFP embeddings for {len(smiles)} SMILES strings..."
        )
        df["ecfp"] = df["smiles"].parallel_apply(smiles_to_ecfp)

        return dict(zip(df["smiles"], df["ecfp"]))

    def embed(self):
        with open(self.INPUT_FILE_PATH, "rb") as f:
            smiles = pickle.load(f)

        # If it's a list of dicts, extract the 'smiles' value from each dict
        if isinstance(smiles, list):
            if len(smiles) > 0 and isinstance(smiles[0], dict):
                self.logger.info(
                    "Detected list of dictionaries, extracting 'smiles' values"
                )
                try:
                    smiles = [d["smiles"] for d in smiles]
                except Exception as e:
                    self.logger.error(
                        f"Error extracting 'smiles' from dictionaries: {e}"
                    )
                    raise ValueError(
                        "Input sequences must be a list of strings or a list of dictionaries with 'smiles' keys."
                    )

            elif len(smiles) > 0 and not isinstance(smiles[0], str):
                self.logger.error(
                    f"Unsupported smiles format. Expected list of strings or list of dicts with 'smiles' keys., got list of {type(smiles[0])}"
                )
                raise ValueError(
                    f"Unsupported smiles format: Expected list of strings or list of dicts with 'smiles' keys., got {type(smiles[0])}"
                )

        elif not isinstance(smiles, list):
            self.logger.error(
                f"Unsupported smiles file format: {type(smiles)}. Expected list of strings or list of dicts with 'smiles' keys.."
            )
            raise ValueError(
                f"Unsupported smiles file format: {type(smiles)} . Expected list of strings or list of dicts with 'smiles' keys.."
            )
        self.logger.info(f"Loaded {len(smiles)} smiles from {self.INPUT_FILE_PATH}")

        existing_embeddings, existing_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.warning(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_embeddings)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        smiles = list(set(smiles) - existing_smiles)
        if len(smiles) == 0:
            self.logger.info("All SMILES embeddings already exist. Exiting.")
            return
        self.logger.info(f"Computing embeddings for {len(smiles)} new smiles")

        try:
            smiles_ecfp_embeddings = self.get_smiles_ecfp_embeddings(smiles)
        except Exception as e:
            self.logger.error(f"Error during ECFP embedding generation: {e}")
            raise e

        smiles_ecfp_embeddings.update(existing_embeddings)
        with open(self.OUTPUT_FILE_PATH, "wb") as f:
            pickle.dump(smiles_ecfp_embeddings, f)
        self.logger.info(
            f"Saved {len(smiles_ecfp_embeddings)} embeddings to {self.OUTPUT_FILE_PATH}"
        )


class SmilesAtomPairFPEmbedder:
    def __init__(self, cfg: DictConfig):
        self.ATOM_PAIR_NBITS = cfg.embeddings.atom_pair_nbits or 1024

        self.PANDARALLEL_NB_WORKERS = (
            cfg.embeddings.atom_pair_pandarallel.nb_workers
            if cfg.embeddings.atom_pair_pandarallel.nb_workers
            else os.cpu_count()
        )
        self.PANDARALLEL_PROGRESS_BAR = (
            cfg.embeddings.atom_pair_pandarallel.progress_bar
            if cfg.embeddings.atom_pair_pandarallel.progress_bar is not None
            else True
        )

        LOG_PATH = Path(cfg.embeddings.log_dir)
        self.INPUT_FILE_PATH = Path(
            cfg.embeddings.smiles_atom_pair_embeddings_input_pkl_path
        )
        self.OUTPUT_FILE_PATH = Path(
            cfg.embeddings.smiles_atom_pair_embeddings_output_pkl_path
        )

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.embeddings.smiles_atom_pair_embeddings_log_file_name,
        ).get_logger()

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.INPUT_FILE_PATH.exists():
            self.logger.error(f"Input file {self.INPUT_FILE_PATH} does not exist.")
            raise FileNotFoundError(
                f"Input file {self.INPUT_FILE_PATH} does not exist."
            )

        for path in [LOG_PATH, self.OUTPUT_FILE_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

    def get_smiles_atom_pair_embeddings(self, smiles):
        from pandarallel import pandarallel
        from src.utils.chem_utils import ChemUtils

        chem_utils = ChemUtils()
        smiles_to_atom_pair_fp = chem_utils.smiles_to_atom_pair_fp
        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )

        df = pd.DataFrame(smiles, columns=["smiles"])
        self.logger.info(
            f"Generating Atom Pair FP embeddings for {len(smiles)} SMILES strings..."
        )
        df["atom_pair_fp"] = df["smiles"].parallel_apply(smiles_to_atom_pair_fp)

        return dict(zip(df["smiles"], df["atom_pair_fp"]))

    def embed(self):
        with open(self.INPUT_FILE_PATH, "rb") as f:
            smiles = pickle.load(f)

        # If it's a list of dicts, extract the 'smiles' value from each dict
        if isinstance(smiles, list):
            if len(smiles) > 0 and isinstance(smiles[0], dict):
                self.logger.info(
                    "Detected list of dictionaries, extracting 'smiles' values"
                )
                try:
                    smiles = [d["smiles"] for d in smiles]
                except Exception as e:
                    self.logger.error(
                        f"Error extracting 'smiles' from dictionaries: {e}"
                    )
                    raise ValueError(
                        "Input sequences must be a list of strings or a list of dictionaries with 'smiles' keys."
                    )

            elif len(smiles) > 0 and not isinstance(smiles[0], str):
                self.logger.error(
                    f"Unsupported smiles format. Expected list of strings or list of dicts with 'smiles' keys., got list of {type(smiles[0])}"
                )
                raise ValueError(
                    f"Unsupported smiles format: Expected list of strings or list of dicts with 'smiles' keys., got {type(smiles[0])}"
                )

        elif not isinstance(smiles, list):
            self.logger.error(
                f"Unsupported smiles file format: {type(smiles)}. Expected list of strings or list of dicts with 'smiles' keys.."
            )
            raise ValueError(
                f"Unsupported smiles file format: {type(smiles)} . Expected list of strings or list of dicts with 'smiles' keys.."
            )
        self.logger.info(f"Loaded {len(smiles)} smiles from {self.INPUT_FILE_PATH}")

        existing_embeddings, existing_smiles = {}, set()
        if self.OUTPUT_FILE_PATH.exists():
            self.logger.warning(
                f"Output file {self.OUTPUT_FILE_PATH} already exists. Finding missing embeddings to compute."
            )
            with open(self.OUTPUT_FILE_PATH, "rb") as f:
                existing_embeddings = pickle.load(f)
                existing_smiles = set(existing_embeddings.keys())
            self.logger.info(
                f"Found {len(existing_embeddings)} existing embeddings in {self.OUTPUT_FILE_PATH}"
            )

        smiles = list(set(smiles) - existing_smiles)
        if len(smiles) == 0:
            self.logger.info("All SMILES embeddings already exist. Exiting.")
            return
        self.logger.info(f"Computing embeddings for {len(smiles)} new smiles")

        try:
            smiles_atom_pair_embeddings = self.get_smiles_atom_pair_embeddings(smiles)
        except Exception as e:
            self.logger.error(f"Error during Atom Pair FP embedding generation: {e}")
            raise e

        smiles_atom_pair_embeddings.update(existing_embeddings)
        with open(self.OUTPUT_FILE_PATH, "wb") as f:
            pickle.dump(smiles_atom_pair_embeddings, f)
        self.logger.info(
            f"Saved {len(smiles_atom_pair_embeddings)} embeddings to {self.OUTPUT_FILE_PATH}"
        )
