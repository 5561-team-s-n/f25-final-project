import argparse
import os
import random
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader

import utils
from dataloader.data_generator import DataGenerator
from dataloader.image_file import ImageFileTrain, ImageFileTest
from dataloader.prefetcher import Prefetcher
from trainers.trainer import Trainer
from utils import CONFIG

torch.manual_seed(8282)
torch.cuda.manual_seed_all(8282)
np.random.seed(8282)
random.seed(8282)


def setup_distributed():
    """Initialize distributed training using torchrun environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        CONFIG.rank = int(os.environ["RANK"])
        CONFIG.local_rank = int(os.environ["LOCAL_RANK"])
        CONFIG.world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(CONFIG.local_rank)

        torch.distributed.init_process_group(
            backend="gloo", #use nccl for linux, gloo for windows"
            init_method="env://"
        )

        CONFIG.distributed = False
        print(f"[Distributed] rank={CONFIG.rank}, local_rank={CONFIG.local_rank}, world_size={CONFIG.world_size}")
    else:
        CONFIG.rank = 0
        CONFIG.local_rank = 0
        CONFIG.world_size = 1
        CONFIG.distributed = False
        print("[Distributed] Running in single-GPU mode.")


def copy_script(root_path=None):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        os.makedirs(CONFIG.log.logging_path)
        os.makedirs(CONFIG.log.checkpoint_path)

    shutil.copytree('./config', os.path.join(root_path, 'config'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./dataloader', os.path.join(root_path, 'dataloader'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./networks', os.path.join(root_path, 'networks'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./trainers', os.path.join(root_path, 'trainers'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('./utils', os.path.join(root_path, 'utils'), ignore=shutil.ignore_patterns('__pycache__'))

    shutil.copy('./main.py', os.path.join(root_path, 'main.py'))
    shutil.copy('./inference.py', os.path.join(root_path, 'inference.py'))
    shutil.copy('./evaluation.py', os.path.join(root_path, 'evaluation.py'))


def main():
    # Setup distributed
    setup_distributed()

    if CONFIG.phase.lower() == "train":

        # Create directories only on rank 0
        if CONFIG.rank == 0:
            utils.make_dir(CONFIG.log.logging_path)
            utils.make_dir(CONFIG.log.checkpoint_path)

        logger = utils.get_logger(CONFIG.log.logging_path, logging_level=CONFIG.log.logging_level)

        # === Build datasets ===
        train_image_file = ImageFileTrain(alpha_dir=CONFIG.data.train_alpha,
                                          fg_dir=CONFIG.data.train_fg,
                                          bg_dir=CONFIG.data.train_bg)
        test_image_file = ImageFileTest(alpha_dir=CONFIG.data.test_alpha,
                                        merged_dir=CONFIG.data.test_merged,
                                        trimap_dir=CONFIG.data.test_trimap)
        train_dataset = DataGenerator(train_image_file, phase='train')
        test_dataset = DataGenerator(test_image_file, phase='val')

        # Distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if CONFIG.distributed else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if CONFIG.distributed else None

        # Dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=CONFIG.model.batch_size,
            shuffle=(train_sampler is None),
            num_workers=CONFIG.data.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
        train_dataloader = Prefetcher(train_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=CONFIG.data.workers,
            sampler=test_sampler,
            drop_last=False
        )

        # Trainer
        trainer = Trainer(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            logger=logger
        )
        trainer.train()

    else:
        raise NotImplementedError(f"Unknown Phase: {CONFIG.phase}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--config', type=str, default='config/MatteFormer_Composition1k.toml')
    args = parser.parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        utils.load_config(toml.load(f))

    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")
    CONFIG.phase = args.phase

    # Experiment dirs
    CONFIG.log.experiment_root = os.path.join(
        CONFIG.log.experiment_root,
        datetime.now().strftime("%y%m%d_%H%M%S")
    )
    CONFIG.log.logging_path = os.path.join(CONFIG.log.experiment_root, CONFIG.log.logging_path)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.experiment_root, CONFIG.log.checkpoint_path)

    # Copy files only on rank 0
    print("CONFIG:")
    pprint(CONFIG)
    copy_script(root_path=CONFIG.log.experiment_root)

    main()
