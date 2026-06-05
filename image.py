from nodes import MAX_RESOLUTION
from .utils_aspect_ratios import avg_from_dims, ar_parts_from_dims, dims_from_ar
import comfy.utils
import torch
import torch.nn.functional as F

class SmartImageResizeAlt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if method == 'stretch':
            new_width = width if width > 0 else ow
            new_height = height if height > 0 else oh

        elif method == 'keep proportion':
            avg = avg_from_dims(width, height)
            n, d = ar_parts_from_dims(ow, oh)
            width, height = dims_from_ar(avg, n, d, multiple_of)

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height

        elif method == 'pad':
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        elif method.startswith('fill'):
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
        else:
            raise ValueError(f"Unknown method: {method}")

        width = new_width
        height = new_height

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],)


import torch
import comfy.utils


class MergeImageBatchList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "overlap": ("INT", {
                    "default": 13,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "overlap_side": (
                    ["source", "new_images"],
                    {"default": "source"}
                ),
                "overlap_mode": (
                    [
                        "cut",
                        "linear_blend",
                        "ease_in_out",
                        "filmic_crossfade",
                        "perceptual_crossfade",
                    ],
                    {"default": "linear_blend"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "image/batch"

    # CRITICAL: this is what makes Comfy pass list-of-batches
    INPUT_IS_LIST = True

    # optional but good practice
    OUTPUT_IS_LIST = (False,)

    # -------------------------
    # core merge logic
    # -------------------------
    @staticmethod
    def merge_batches(
        source_images,
        new_images,
        overlap,
        overlap_side,
        overlap_mode,
    ):
        if source_images.shape[1:3] != new_images.shape[1:3]:
            raise ValueError(
                f"Source and new images must have same shape: "
                f"{source_images.shape[1:3]} vs {new_images.shape[1:3]}"
            )

        overlap = min(overlap, len(source_images), len(new_images))

        if overlap <= 0:
            return torch.cat((source_images, new_images), dim=0)

        prefix = source_images[:-overlap]

        if overlap_side == "source":
            blend_src = source_images[-overlap:]
            blend_dst = new_images[:overlap]
        else:
            blend_src = new_images[:overlap]
            blend_dst = source_images[-overlap:]

        suffix = new_images[overlap:]

        # -------------------------
        # linear blend
        # -------------------------
        if overlap_mode == "linear_blend":
            alpha = torch.linspace(
                0, 1, overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype
            )[1:-1].view(-1, 1, 1, 1)

            blended = (1 - alpha) * blend_src + alpha * blend_dst
            return torch.cat((prefix, blended, suffix), dim=0)

        # -------------------------
        # ease in/out
        # -------------------------
        elif overlap_mode == "ease_in_out":
            t = torch.linspace(
                0, 1, overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype
            )[1:-1]

            eased = (3 * t * t - 2 * t * t * t).view(-1, 1, 1, 1)

            blended = (1 - eased) * blend_src + eased * blend_dst
            return torch.cat((prefix, blended, suffix), dim=0)

        # -------------------------
        # filmic crossfade
        # -------------------------
        elif overlap_mode == "filmic_crossfade":
            gamma = 2.2

            alpha = torch.linspace(
                0, 1, overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype
            )[1:-1].view(-1, 1, 1, 1)

            src = torch.pow(blend_src, gamma)
            dst = torch.pow(blend_dst, gamma)

            blended = (1 - alpha) * src + alpha * dst
            blended = torch.pow(blended, 1.0 / gamma)

            return torch.cat((prefix, blended, suffix), dim=0)

        # -------------------------
        # perceptual crossfade
        # -------------------------
        elif overlap_mode == "perceptual_crossfade":
            import kornia

            alpha = torch.linspace(
                0, 1, overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype
            )[1:-1].view(-1, 1, 1, 1)

            src = blend_src.movedim(-1, 1)
            dst = blend_dst.movedim(-1, 1)

            lab_src = kornia.color.rgb_to_lab(src)
            lab_dst = kornia.color.rgb_to_lab(dst)

            blended = (1 - alpha) * lab_src + alpha * lab_dst
            blended = kornia.color.lab_to_rgb(blended)

            blended = blended.movedim(1, -1)

            return torch.cat((prefix, blended, suffix), dim=0)

        # -------------------------
        # cut
        # -------------------------
        elif overlap_mode == "cut":
            if overlap_side == "new_images":
                return torch.cat(
                    (source_images, new_images[overlap:]),
                    dim=0,
                )

            return torch.cat(
                (source_images[:-overlap], new_images),
                dim=0,
            )

        raise ValueError(f"Unknown overlap mode: {overlap_mode}")

    # -------------------------
    # execute over list of batches
    # -------------------------
    def execute(self, images, overlap, overlap_side, overlap_mode):

        # -------------------------
        # unwrap Comfy list-wrapped scalars
        # -------------------------
        if isinstance(overlap, list):
            overlap = overlap[0]

        if isinstance(overlap_side, list):
            overlap_side = overlap_side[0]

        if isinstance(overlap_mode, list):
            overlap_mode = overlap_mode[0]

        if not images:
            raise ValueError("No image batches supplied")

        if len(images) == 1:
            return (images[0],)

        merged = images[0]

        for batch in images[1:]:
            merged = self.merge_batches(
                merged,
                batch,
                overlap,
                overlap_side,
                overlap_mode,
            )

        return (merged,)

IMAGE_CLASS_MAPPINGS = {
    "SmartImageResizeAlt": SmartImageResizeAlt,
    "MergeImageBatchList": MergeImageBatchList,
}

IMAGE_NAME_MAPPINGS = {
    "SmartImageResizeAlt": "🔧 Smart Image Resize Alt ◯",
    "MergeImageBatchList": "🔧 MergeImageBatchList ◯",
}
