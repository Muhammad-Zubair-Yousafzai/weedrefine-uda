"""SAM 2 automatic masks to a single region id map."""
import numpy as np


def masks_to_region_map(masks, shape):
    """Convert SAM mask dicts (with 'segmentation' and 'area') to an id map.

    Larger masks are painted first so smaller ones (often weeds) stay on top
    where masks overlap. Pixels covered by no mask get -1.
    """
    region_map = np.full(shape, -1, dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda i: masks[i]["area"], reverse=True)
    for new_id, i in enumerate(order):
        region_map[masks[i]["segmentation"].astype(bool)] = new_id
    return region_map


class SamRegionGenerator:
    """Wrapper around SAM 2's automatic mask generator. Needs a GPU for practical speed."""

    def __init__(self, checkpoint, model_cfg, device="cuda", **amg_kwargs):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(model, **amg_kwargs)

    def __call__(self, image_rgb):
        masks = self.generator.generate(image_rgb)
        return masks_to_region_map(masks, image_rgb.shape[:2])
