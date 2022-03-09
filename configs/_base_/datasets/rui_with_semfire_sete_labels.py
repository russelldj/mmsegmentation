# dataset settings
dataset_type = "CustomDataset"
# TODO Consider having a different one for the dif
data_root = "data/rui_with_semfire_sete_labels/"
classes = (
    "background",
    "fuel",
    "trunks",
    "humans",
    "animal",
    "canopies",
    "traversable",
)

# TODO update
# label_map = {0:0, 1:1}
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (256, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(848, 480), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(848, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img_dir/train",
        ann_dir="ann_dir/train",
        pipeline=train_pipeline,
        img_suffix="_rgb.png",
        seg_map_suffix="_segmentation.png",
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img_dir/val",
        ann_dir="ann_dir/val",
        pipeline=test_pipeline,
        img_suffix="_rgb.png",
        seg_map_suffix="_segmentation.png",
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img_dir/test",
        ann_dir="ann_dir/test",
        img_suffix="_rgb.png",
        seg_map_suffix="_segmentation.png",
        pipeline=test_pipeline,
        classes=classes,
    ),
)
