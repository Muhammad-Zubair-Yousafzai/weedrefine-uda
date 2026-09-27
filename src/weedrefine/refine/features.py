"""Dense patch features from a DINOv3 (or DINOv2) backbone via Hugging Face transformers."""
import cv2


def grid_size(h, w, long_side, patch):
    """Input size that keeps the aspect ratio, long side ~long_side, both sides multiples of patch."""
    s = long_side / max(h, w)
    return (max(patch, round(h * s / patch) * patch),
            max(patch, round(w * s / patch) * patch))


class DenseFeatureExtractor:
    def __init__(self, model_name, device="cuda", long_side=1024):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.long_side = long_side
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch = getattr(self.model.config, "patch_size", 16)
        # DINOv3 prepends CLS plus register tokens; DINOv2 has CLS only.
        self.n_prefix = 1 + getattr(self.model.config, "num_register_tokens", 0)

    def __call__(self, image_rgb):
        """Return (h/patch, w/patch, D) patch features for the aspect-preserving resized image.

        Features stay at grid resolution; merge.region_mean_features maps regions onto the grid.
        The processor's default 224x224 square resize is skipped on purpose.
        """
        torch = self.torch
        h, w = grid_size(*image_rgb.shape[:2], self.long_side, self.patch)
        resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        inputs = self.processor(images=resized, do_resize=False, do_center_crop=False,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            tokens = self.model(**inputs).last_hidden_state[:, self.n_prefix:]
        ph = inputs["pixel_values"].shape[-2] // self.patch
        pw = inputs["pixel_values"].shape[-1] // self.patch
        return tokens.reshape(ph, pw, -1).float().cpu().numpy()
