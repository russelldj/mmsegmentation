# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import mmcv
import os
import os.path as osp
import time
import torch
import warnings
from argparse import Namespace
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from sacred import Experiment
from sacred.observers import MongoObserver

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

ex = Experiment("train_mmseg")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    config = None  # train config file path
    work_dir = None  # the dir to save logs and models
    load_from = None  # the checkpoint file to load weights from
    resume_from = None  # the checkpoint file to resume from
    no_validate = False  # whether not to evaluate the checkpoint during training
    gpus = None  # number of gpus to use (only applicable to non-distributed training)
    gpu_ids = None  # ids of gpus to use (only applicable to non-distributed training)
    seed = None  # random seed
    deterministic = False  # whether to set deterministic options for CUDNN backend

    # TODO
    # Conisder adding "--options" and "--cfg-options"
    launcher = (
        "none"  # choices=["none", "pytorch", "slurm", "mpi"], help="job launcher",
    )
    local_rank = 0

    args = Namespace(
        config=config,
        work_dir=work_dir,
        load_from=load_from,
        resume_from=resume_from,
        no_validate=no_validate,
        gpus=gpus,
        gpu_ids=gpu_ids,
        seed=seed,
        deterministic=deterministic,
        launcher=launcher,
        local_rank=local_rank,
    )
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)


@ex.automain
def main(args):

    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #    cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            "SyncBN is only supported with DDP. To be compatible with DP, "
            "we convert SyncBN to BN. Please use dist_train.sh which can "
            "avoid this error."
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f"{__version__}+{get_git_hash()[:7]}",
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )
