# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Safeforest23Dataset(BaseSegDataset):
    """Safeforest 2023 dataset.
    """
    METAINFO = dict(
        classes = (
            "Dry Grass",
            "Green Grass",
            "Dry Shrubs",
            "Green Shrubs",
            "Canopy",
            "Wood Pieces",
            "Litterfall",
            "Timber Litter",
            "Live Trunks",
            "Bare Earth",
            "People",
            "Sky",
            "Blurry",
            "Obstacles",
            "Drones",
        ),
        palette=[
            [128, 224, 255],  # Dry Grass, 0
            [0, 255, 255],  # Green Grass (canopy), 1
            [80, 0, 255],  # Dry Shrubs, 2
            [45, 112, 134],  # Green Shrubs, 3
            [0, 255, 144],  # Canopy, 4
            [128, 255, 199],  # Wood Pieces, 5
            [224, 0, 255],  # Litterfall (bare earth or fuel), 6
            [0, 194, 255],  # Timber Litter, 7
            [45, 134, 95],  # Live Trunks, 8
            [255, 0, 111],  # Bare Earth, 9
            [239, 128, 255],  # People, 10
            [167, 128, 255],  # Sky, 11
            [134, 45, 83],  # Blurry, 12
            [83, 45, 134],  # Obstacle
            [45, 68, 134]])  # Drones, 13

    def __init__(self,
                 img_suffix='_rgb.png',
                 seg_map_suffix='_segmentation.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
