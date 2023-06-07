# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Safeforest23CompressedDataset(BaseSegDataset):
    """Safeforest 2023 dataset.
    """
    METAINFO = dict(
        classes = (
            "Fuel",
            "Canopy",
            "Background",
            "Trunks",
        ),
        palette=[
            [255, 0, 0],  # Fuel
            [0, 255, 0],  # Canopy
            [0, 0, 0],  # Background
            [255, 0, 255]  # Trunks
            ])

    def __init__(self,
                 img_suffix='_rgb.png',
                 seg_map_suffix='_segmentation.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
