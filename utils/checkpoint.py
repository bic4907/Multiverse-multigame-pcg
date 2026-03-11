import os
import re
from glob import glob
import torch
from typing import Optional

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


class CheckpointManager:
    """Reusable checkpoint manager.

    This version keeps a minimal constructor and lets users register models,
    optimizers, and schedulers via `register_*` methods. Saving will iterate
    registered items and store their state_dicts.
    """

    def __init__(
        self,
        log_dir_root,
        config=None,
        save_interval: Optional[int] = None,
        save_keep: Optional[int] = None,
        logger=None,
        get_log_dir_fn=None,
    ):
        self.log_dir_root = log_dir_root
        self.config = config
        self.logger = logger
        self.get_log_dir_fn = get_log_dir_fn

        # containers for registered objects
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}

        # defaults if not specified
        self.save_interval = int(save_interval) if save_interval is not None else int(getattr(config, 'save_interval', 5) if config is not None else 5)
        self.save_keep = int(save_keep) if save_keep is not None else int(getattr(config, 'save_keep', 3) if config is not None else 3)

    @staticmethod
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: CheckpointManager._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CheckpointManager._to_cpu(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(CheckpointManager._to_cpu(x) for x in obj)
        return obj

    def _prepare_optimizer_state(self, opt):
        sd = opt.state_dict()
        if 'state' in sd and isinstance(sd['state'], dict):
            sd['state'] = {k: CheckpointManager._to_cpu(v) for k, v in sd['state'].items()}
        return sd

    def _serialize_config(self):
        if OmegaConf is not None and self.config is not None:
            try:
                return OmegaConf.to_container(self.config, resolve=True)
            except Exception:
                pass
        # fallback
        try:
            return dict(self.config) if self.config is not None else {}
        except Exception:
            return str(self.config)

    # registration helpers
    def register(self, **models):
        """Register models to be saved. Usage: register_models(clip=clip_model, vae=vae_model)"""
        for k, v in models.items():
            self.models[k] = v

    def save(self, epoch: int):
        # Save actual files in global checkpoints dir, create symlinks in epoch dirs
        global_ckpt_dir = os.path.join(self.log_dir_root, "checkpoints")
        os.makedirs(global_ckpt_dir, exist_ok=True)

        # Actual checkpoint file in global dir
        ckpt_path = os.path.join(global_ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
        tmp_path = ckpt_path + ".tmp"

        # Prepare CPU-safe states by iterating registered components
        model_states = {name: CheckpointManager._to_cpu(m.state_dict()) for name, m in self.models.items()}

        optimizer_states = {}
        for name, opt in self.optimizers.items():
            try:
                optimizer_states[name] = self._prepare_optimizer_state(opt)
            except Exception:
                optimizer_states[name] = None

        scheduler_states = {}
        for name, sch in self.schedulers.items():
            try:
                if hasattr(sch, 'state_dict'):
                    scheduler_states[name] = CheckpointManager._to_cpu(sch.state_dict())
                else:
                    scheduler_states[name] = None
            except Exception:
                scheduler_states[name] = None

        checkpoint = {
            'epoch': epoch,
            'config': self._serialize_config(),
            'models': model_states,
            'optimizers': optimizer_states,
            'schedulers': scheduler_states,
        }

        # atomic save to global checkpoints dir
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, ckpt_path)
            if self.logger:
                self.logger.info(f"Saved checkpoint to {ckpt_path}")
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Failed to save checkpoint to {ckpt_path}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return  # don't create symlink if save failed

        # Create symlink in epoch-specific dir pointing to global checkpoint
        try:
            if self.get_log_dir_fn is None:
                epoch_ckpt_dir = os.path.join(self.log_dir_root, f"epoch_{epoch}", "checkpoints")
            else:
                epoch_ckpt_dir = self.get_log_dir_fn(root_dir=self.log_dir_root, epoch=epoch, sub_dir="checkpoints")

            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            epoch_link = os.path.join(epoch_ckpt_dir, f"checkpoint_epoch_{epoch}.pt")

            # remove existing link/file if present
            try:
                if os.path.islink(epoch_link) or os.path.exists(epoch_link):
                    os.remove(epoch_link)
            except Exception:
                pass

            # create symlink using relative path
            try:
                rel_path = os.path.relpath(ckpt_path, epoch_ckpt_dir)
                os.symlink(rel_path, epoch_link)
            except Exception:
                # fall back to copy if symlink not supported
                try:
                    import shutil
                    shutil.copy2(ckpt_path, epoch_link)
                except Exception:
                    if self.logger:
                        self.logger.warning(f"Failed to create epoch symlink/copy for {ckpt_path}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create epoch checkpoint link: {e}")

        # rotation: keep last N checkpoint files in global dir
        try:
            if self.save_keep > 0:
                pattern_all = os.path.join(global_ckpt_dir, 'checkpoint_epoch_*.pt')
                all_files = glob(pattern_all)

                def _parse_epoch(p):
                    m = re.search(r'checkpoint_epoch_(\d+)\.pt$', p)
                    return int(m.group(1)) if m else -1

                files_sorted = sorted(all_files, key=lambda p: _parse_epoch(p))
                if len(files_sorted) > self.save_keep:
                    to_remove = files_sorted[:-self.save_keep]
                    for p in to_remove:
                        try:
                            removed_epoch = _parse_epoch(p)
                            # remove the actual checkpoint file
                            os.remove(p)
                            if self.logger:
                                self.logger.info(f"Removed old checkpoint: {p}")

                            # also remove corresponding epoch dir symlink
                            if self.get_log_dir_fn is None:
                                epoch_dir = os.path.join(self.log_dir_root, f"epoch_{removed_epoch}", "checkpoints")
                            else:
                                epoch_dir = self.get_log_dir_fn(root_dir=self.log_dir_root, epoch=removed_epoch, sub_dir="checkpoints")

                            epoch_link = os.path.join(epoch_dir, os.path.basename(p))
                            try:
                                if os.path.islink(epoch_link) or os.path.exists(epoch_link):
                                    os.remove(epoch_link)
                            except Exception:
                                pass

                            # try to remove empty dirs
                            try:
                                if os.path.isdir(epoch_dir) and not os.listdir(epoch_dir):
                                    os.rmdir(epoch_dir)
                                    parent_epoch_dir = os.path.dirname(epoch_dir)
                                    if os.path.isdir(parent_epoch_dir) and not os.listdir(parent_epoch_dir):
                                        os.rmdir(parent_epoch_dir)
                            except Exception:
                                pass
                        except Exception:
                            if self.logger:
                                self.logger.warning(f"Failed to remove old checkpoint: {p}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Checkpoint rotation failed: {e}")

    def maybe_save(self, epoch: int, is_last_epoch: bool = False):
        # epoch is 1-based here
        if (self.save_interval > 0 and (epoch % self.save_interval) == 0) or is_last_epoch:
            self.save(epoch)

    # -------------------- Load / restore helpers --------------------
    def _checkpoint_path_for_epoch(self, epoch: int):
        # checkpoints are now stored in global checkpoints dir
        return os.path.join(self.log_dir_root, "checkpoints", f"checkpoint_epoch_{epoch}.pt")

    def _find_latest_checkpoint(self):
        # find in global checkpoints dir
        pattern = os.path.join(self.log_dir_root, 'checkpoints', 'checkpoint_epoch_*.pt')
        files = glob(pattern)
        def _parse_epoch(p):
            m = re.search(r'checkpoint_epoch_(\d+)\.pt$', p)
            return int(m.group(1)) if m else -1
        if not files:
            return None
        latest = max(files, key=lambda p: _parse_epoch(p))
        return latest

    def save_models(self, epoch: int, names: Optional[list] = None):
        """Save only the registered models' state_dicts for the given epoch.

        Writes a models-only checkpoint file named `checkpoint_models_epoch_{epoch}.pt`
        under the same checkpoints folder.
        """
        if names is None:
            names = list(self.models.keys())
        # make ckpt dir
        if self.get_log_dir_fn is None:
            ckpt_dir = os.path.join(self.log_dir_root, f"epoch_{epoch}", "checkpoints")
        else:
            ckpt_dir = self.get_log_dir_fn(root_dir=self.log_dir_root, epoch=epoch, sub_dir="checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_models_epoch_{epoch}.pt")
        tmp_path = ckpt_path + ".tmp"

        model_states = {}
        for name in names:
            m = self.models.get(name)
            if m is None:
                if self.logger:
                    self.logger.warning(f"Model '{name}' not registered; skipping save.")
                continue
            try:
                model_states[name] = CheckpointManager._to_cpu(m.state_dict())
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to get state_dict for model '{name}': {e}")

        checkpoint = {'epoch': epoch, 'models': model_states}

        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, ckpt_path)
            if self.logger:
                self.logger.info(f"Saved models-only checkpoint to {ckpt_path}")
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Failed to save models-only checkpoint to {ckpt_path}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _find_latest_models_checkpoint(self):
        pattern = os.path.join(self.log_dir_root, 'epoch_*', 'checkpoints', 'checkpoint_models_epoch_*.pt')
        files = glob(pattern)
        def _parse_epoch(p):
            m = re.search(r'checkpoint_models_epoch_(\d+)\.pt$', p)
            return int(m.group(1)) if m else -1
        if not files:
            return None
        latest = max(files, key=lambda p: _parse_epoch(p))
        return latest

    def load_models(self, epoch: Optional[int] = None, path: Optional[str] = None, map_location: Optional[str] = 'cpu',
                    names: Optional[list] = None):
        """Load models-only checkpoint and restore registered models.

        If `path` specified, loads from path; elif `epoch` specified, loads models-only file for that epoch;
        otherwise tries to find the latest models-only checkpoint. Returns dict with path and restored names.

        Will also try to load from full checkpoint files if models-only files are not found.
        """
        ckpt_path = None
        if path is not None:
            ckpt_path = path
        elif epoch is not None:
            if self.get_log_dir_fn is None:
                ckpt_path = os.path.join(self.log_dir_root, f"epoch_{epoch}", "checkpoints", f"checkpoint_models_epoch_{epoch}.pt")
            else:
                ckpt_dir = self.get_log_dir_fn(root_dir=self.log_dir_root, epoch=epoch, sub_dir="checkpoints")
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_models_epoch_{epoch}.pt")

            # If models-only checkpoint doesn't exist, try full checkpoint
            if not os.path.exists(ckpt_path):
                full_ckpt_path = self._checkpoint_path_for_epoch(epoch)
                if os.path.exists(full_ckpt_path):
                    ckpt_path = full_ckpt_path
                    if self.logger:
                        self.logger.info(f"Models-only checkpoint not found, using full checkpoint: {ckpt_path}")
        else:
            # Try models-only first, then fall back to full checkpoint
            ckpt_path = self._find_latest_models_checkpoint()
            if not ckpt_path or not os.path.exists(ckpt_path):
                ckpt_path = self._find_latest_checkpoint()
                if ckpt_path and self.logger:
                    self.logger.info(f"Models-only checkpoint not found, using full checkpoint: {ckpt_path}")

        if not ckpt_path or not os.path.exists(ckpt_path):
            if self.logger:
                self.logger.warning(f"No checkpoint found to load (epoch={epoch}, path={path})")
            return {'path': None, 'loaded': {}}

        ckpt = torch.load(ckpt_path, map_location=map_location)
        loaded = {}

        if not isinstance(ckpt, dict) or 'models' not in ckpt:
            if self.logger:
                self.logger.warning(f"Checkpoint at {ckpt_path} does not contain models key")
            return {'path': ckpt_path, 'loaded': {}}

        model_dict = ckpt['models']
        if names is None:
            names = list(model_dict.keys())

        for name in names:
            if name not in model_dict:
                if self.logger:
                    self.logger.warning(f"No state for model '{name}' in checkpoint {ckpt_path}")
                continue
            model = self.models.get(name)
            if model is None:
                if self.logger:
                    self.logger.warning(f"Model '{name}' not registered; cannot restore")
                continue
            try:
                model.load_state_dict(model_dict[name])
                loaded.setdefault('models', []).append(name)
            except Exception:
                try:
                    model.load_state_dict(model_dict[name], strict=False)
                    loaded.setdefault('models', []).append(name)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load model '{name}' from {ckpt_path}: {e}")

        if self.logger:
            self.logger.info(f"Loaded checkpoint from {ckpt_path}; restored models: {loaded.get('models', [])}")

        return {'path': ckpt_path, 'loaded': loaded}

class DummyCheckpointManager:
    """A dummy checkpoint manager that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def register(self, **models):
        pass

    def save(self, epoch: int):
        pass

    def maybe_save(self, epoch: int, is_last_epoch: bool = False):
        pass

    def save_models(self, epoch: int, names: Optional[list] = None):
        pass

    def load_models(self, epoch: Optional[int] = None, path: Optional[str] = None, map_location: Optional[str] = 'cpu',
                    names: Optional[list] = None):
        return {'path': None, 'loaded': {}}


def load_checkpoint_with_fallback(
    checkpoint_manager: CheckpointManager,
    config,
    device: str,
    model_names: list,
    logger=None
):
    """
    Reusable checkpoint loading utility with fallback logic.

    Priority order: checkpoint_path > checkpoint_epoch > latest from exp_path

    Args:
        checkpoint_manager: CheckpointManager instance with registered models
        config: Config object that may have checkpoint_path or checkpoint_epoch attributes
        device: Device to load checkpoint to (e.g., 'cuda', 'cpu')
        model_names: List of model names to load (e.g., ['clip', 'vae'])
        logger: Logger instance for logging messages

    Returns:
        dict: Result dictionary with 'path' and 'loaded' keys

    Example:
        ```python
        # Create checkpoint manager and register models
        checkpoint_manager = CheckpointManager(
            log_dir_root=config.exp_path,
            config=config,
            logger=logger,
            get_log_dir_fn=get_log_dir,
        )
        checkpoint_manager.register(clip=clip_model, vae=vae_model)

        # Load checkpoint with automatic fallback
        load_checkpoint_with_fallback(
            checkpoint_manager=checkpoint_manager,
            config=config,
            device=config.device,
            model_names=['clip', 'vae'],
            logger=logger
        )
        ```
    """
    if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
        # Load from specific path
        if logger:
            logger.info(f"Loading checkpoint from: {config.checkpoint_path}")
        result = checkpoint_manager.load_models(
            path=config.checkpoint_path,
            map_location=device,
            names=model_names
        )
        if result['path']:
            if logger:
                logger.info(f"Successfully loaded checkpoint: {result['loaded']}")
        else:
            if logger:
                logger.warning(f"Failed to load checkpoint from {config.checkpoint_path}")

    elif hasattr(config, 'checkpoint_epoch') and config.checkpoint_epoch is not None:
        # Load from specific epoch
        if logger:
            logger.info(f"Loading checkpoint from epoch: {config.checkpoint_epoch}")
        result = checkpoint_manager.load_models(
            epoch=config.checkpoint_epoch,
            map_location=device,
            names=model_names
        )
        if result['path']:
            if logger:
                logger.info(f"Successfully loaded checkpoint: {result['loaded']}")
        else:
            if logger:
                logger.warning(f"Failed to load checkpoint for epoch {config.checkpoint_epoch}")

    else:
        # Auto-load latest checkpoint from exp_path
        if logger:
            logger.info(f"Auto-loading latest checkpoint from {config.exp_path}...")
        result = checkpoint_manager.load_models(
            map_location=device,
            names=model_names
        )
        if result['path']:
            if logger:
                logger.info(f"Successfully loaded latest checkpoint: {result['loaded']} from {result['path']}")
        else:
            if logger:
                logger.warning(f"No checkpoint found in {config.exp_path}, using randomly initialized models")

    return result
