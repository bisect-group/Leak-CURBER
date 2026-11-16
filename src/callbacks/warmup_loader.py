from lightning.pytorch.callbacks import Callback

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WarmupLoaderCallback(Callback):
    """
    Callback to switch from warmup (sorted) dataloader to normal dataloader.

    Only reloads the dataloader once at the transition point, preserving
    persistent_workers benefits for most of training.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm is None:
            return

        warmup_epochs = getattr(dm.hparams, "warmup_epochs", 0)
        use_warmup = getattr(dm.hparams, "use_warmup_loader", False)

        if not use_warmup or warmup_epochs <= 0:
            return

        # Reload dataloader only at the transition point
        if trainer.current_epoch == warmup_epochs:
            log.info(
                f"Epoch {trainer.current_epoch}: Switching from warmup to normal loader."
            )
            trainer.fit_loop._combined_loader = None
            trainer.fit_loop.setup_data()
