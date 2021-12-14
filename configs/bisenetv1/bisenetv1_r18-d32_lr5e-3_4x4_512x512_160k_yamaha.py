_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/yamaha.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=8),
    auxiliary_head=[
        dict(num_classes=8),
        dict(num_classes=8),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)
