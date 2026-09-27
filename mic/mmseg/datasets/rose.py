# ROSE crop/weed dataset for MIC (MMSegmentation 0.16 style).
# Copied into third_party/MIC/seg/mmseg/datasets/ by scripts/install_mic_rose.sh.
# Layout written by scripts/prepare_mic_data.py:
#   {data_root}/img/{stem}.jpg, {data_root}/ann/{stem}.png (0 bg, 1 crop, 2 weed, 255 ignore)

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ROSEDataset(CustomDataset):
    CLASSES = ('background', 'crop', 'weed')
    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, img_suffix='.jpg', **kwargs):
        super(ROSEDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix='.png', **kwargs)
