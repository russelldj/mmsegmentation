_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/semfire_sete_train_15.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_finetune_10k.py",
]

model = dict(pretrained="pretrain/mit_b0.pth", decode_head=dict(num_classes=7))

# optimizer
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
