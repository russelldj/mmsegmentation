_base_ = ["./segformer_mit-b0_512x512_10k_sete_20_finetune.py"]

# model settings
model = dict(
    pretrained="/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth",
    backbone=dict(embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
)
