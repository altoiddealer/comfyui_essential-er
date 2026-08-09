from comfy_api.latest import io

import comfy.utils
import torch
import torch.nn.functional as F

from .utils_aspect_ratios import (
    avg_from_dims,
    ar_parts_from_dims,
    dims_from_ar,
)


MAX_RESOLUTION = 16384


class SmartImageResizeAlt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SmartImageResizeAlt",
            display_name="🔧 Smart Image Resize Alt ◯",
            description="Resize an image using multiple methods while optionally preserving aspect ratio and enforcing dimension multiples.",
            category="essentials/image manipulation",
            search_aliases=[
                "resize",
                "resize image",
                "smart resize",
                "image resize",
                "scale",
            ],
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to resize.",
                ),

                io.Int.Input(
                    "width",
                    default=512,
                    min=0,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Target width in pixels. Set to 0 to automatically use the source width.",
                ),

                io.Int.Input(
                    "height",
                    default=512,
                    min=0,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Target height in pixels. Set to 0 to automatically use the source height.",
                ),

                io.Combo.Input(
                    "interpolation",
                    options=[
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                        "lanczos",
                    ],
                    default="nearest",
                    tooltip="Interpolation method used when resizing.",
                ),

                io.Combo.Input(
                    "method",
                    options=[
                        "stretch",
                        "keep proportion",
                        "fill / crop",
                        "pad",
                    ],
                    default="stretch",
                    tooltip="How the image should be resized relative to the requested dimensions.",
                ),

                io.Combo.Input(
                    "condition",
                    options=[
                        "always",
                        "downscale if bigger",
                        "upscale if smaller",
                        "if bigger area",
                        "if smaller area",
                    ],
                    default="always",
                    tooltip="Condition controlling whether the resize operation is applied.",
                ),

                io.Int.Input(
                    "multiple_of",
                    default=0,
                    min=0,
                    max=512,
                    step=1,
                    tooltip="If greater than 1, crop the final image so both dimensions are divisible by this value.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                ),
                io.Int.Output(
                    display_name="width",
                ),
                io.Int.Output(
                    display_name="height",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        width,
        height,
        interpolation="nearest",
        method="stretch",
        condition="always",
        multiple_of=0,
    ) -> io.NodeOutput:

        _, oh, ow, _ = image.shape

        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if method == "stretch":
            new_width = width if width > 0 else ow
            new_height = height if height > 0 else oh

        elif method == "keep proportion":
            avg = avg_from_dims(width, height)
            n, d = ar_parts_from_dims(ow, oh)

            width, height = dims_from_ar(
                avg,
                n,
                d,
                multiple_of,
            )

            ratio = max(
                width / ow,
                height / oh,
            )

            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            x = (new_width - width) // 2
            y = (new_height - height) // 2

            x2 = x + width
            y2 = y + height

            if x2 > new_width:
                x -= x2 - new_width

            if x < 0:
                x = 0

            if y2 > new_height:
                y -= y2 - new_height

            if y < 0:
                y = 0

            width = new_width
            height = new_height

        elif method == "pad":
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = min(
                width / ow,
                height / oh,
            )

            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left

            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        elif method == "fill / crop":

            if multiple_of > 0:
                width = width - (width % multiple_of)
                height = height - (height % multiple_of)

            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(
                width / ow,
                height / oh,
            )

            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            x = (new_width - width) // 2
            y = (new_height - height) // 2

            x2 = x + width
            y2 = y + height

            if x2 > new_width:
                x -= x2 - new_width

            if x < 0:
                x = 0

            if y2 > new_height:
                y -= y2 - new_height

            if y < 0:
                y = 0

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        width = new_width
        height = new_height

        should_resize = (
            condition == "always"
            or (
                condition == "downscale if bigger"
                and (oh > height or ow > width)
            )
            or (
                condition == "upscale if smaller"
                and (oh < height or ow < width)
            )
            or (
                condition == "if bigger area"
                and (oh * ow > height * width)
            )
            or (
                condition == "if smaller area"
                and (oh * ow < height * width)
            )
        )

        if should_resize:
            outputs = image.permute(
                0,
                3,
                1,
                2,
            )

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(
                    outputs,
                    width,
                    height,
                )
            else:
                outputs = F.interpolate(
                    outputs,
                    size=(height, width),
                    mode=interpolation,
                )

            if method == "pad":
                if (
                    pad_left > 0
                    or pad_right > 0
                    or pad_top > 0
                    or pad_bottom > 0
                ):
                    outputs = F.pad(
                        outputs,
                        (
                            pad_left,
                            pad_right,
                            pad_top,
                            pad_bottom,
                        ),
                        value=0,
                    )

            outputs = outputs.permute(
                0,
                2,
                3,
                1,
            )

            if method == "fill / crop":
                if (
                    x > 0
                    or y > 0
                    or x2 > 0
                    or y2 > 0
                ):
                    outputs = outputs[
                        :,
                        y:y2,
                        x:x2,
                        :,
                    ]

        else:
            outputs = image

        if (
            multiple_of > 1
            and (
                outputs.shape[2] % multiple_of != 0
                or outputs.shape[1] % multiple_of != 0
            )
        ):
            output_width = outputs.shape[2]
            output_height = outputs.shape[1]

            x = (output_width % multiple_of) // 2
            y = (output_height % multiple_of) // 2

            x2 = output_width - (
                (output_width % multiple_of) - x
            )

            y2 = output_height - (
                (output_height % multiple_of) - y
            )

            outputs = outputs[
                :,
                y:y2,
                x:x2,
                :,
            ]

        outputs = torch.clamp(
            outputs,
            0,
            1,
        )

        return io.NodeOutput(
            outputs,
            outputs.shape[2],
            outputs.shape[1],
        )
