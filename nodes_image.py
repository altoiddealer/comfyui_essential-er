import math

import torch
import torch.nn.functional as F

import comfy.utils
from nodes import MAX_RESOLUTION
from comfy_api.latest import io

from .utils_aspect_ratios import (
    avg_from_dims,
    ar_parts_from_dims,
    dims_from_ar,
)

class ResizeImageMaskAlt(io.ComfyNode):

    # ------------------------------------------------------------------
    # Input / output configuration
    # ------------------------------------------------------------------

    scale_methods = [
        "nearest",
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]

    crop_methods = [
        "disabled",
        "center",
    ]

    legacy_methods = [
        "keep proportion",
        "pad",
    ]

    pad_colors = [
        "black",
        "gray",
        "white",
    ]

    conditions = [
        "always",
        "downscale if bigger",
        "upscale if smaller",
        "if bigger area",
        "if smaller area",
    ]

    # ------------------------------------------------------------------
    # General dimension helpers
    # ------------------------------------------------------------------

    @staticmethod
    def round_to_multiple(value, multiple):
        """
        Round a dimension to the nearest multiple.
        """
        value = int(round(value))

        if multiple <= 1:
            return max(1, value)

        return max(
            multiple,
            int(round(value / multiple)) * multiple,
        )

    @classmethod
    def round_dimensions_to_multiple(
        cls,
        width,
        height,
        multiple,
    ):
        """
        Round both dimensions independently to the nearest multiple.
        """
        if multiple <= 1:
            return (
                max(1, int(round(width))),
                max(1, int(round(height))),
            )

        return (
            cls.round_to_multiple(width, multiple),
            cls.round_to_multiple(height, multiple),
        )

    @staticmethod
    def floor_to_multiple(value, multiple):
        """
        Floor a dimension to a multiple.
        """
        value = int(value)

        if multiple <= 1:
            return max(1, value)

        return max(
            multiple,
            (value // multiple) * multiple,
        )

    # ------------------------------------------------------------------
    # Image / Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_image(input_tensor):
        """
        IMAGE:
            [B, H, W, C]

        MASK:
            [B, H, W]
        """
        return len(input_tensor.shape) == 4

    @staticmethod
    def init_image_mask_input(
        input_tensor,
        is_type_image,
    ):
        """
        Convert either IMAGE or MASK to:

            [B, C, H, W]
        """
        if is_type_image:
            return input_tensor.movedim(-1, 1)

        return input_tensor.unsqueeze(1)

    @staticmethod
    def finalize_image_mask_input(
        input_tensor,
        is_type_image,
    ):
        """
        Convert [B, C, H, W] back to the original IMAGE/MASK layout.
        """
        if is_type_image:
            return input_tensor.movedim(1, -1)

        return input_tensor.squeeze(1)

    @staticmethod
    def pad_color_value(color):
        """
        Convert a pad color name to the normalized tensor value used by
        ComfyUI IMAGE and MASK tensors.
        """

        values = {
            "black": 0.0,
            "gray": 0.5,
            "white": 1.0,
        }

        try:
            return values[color]
        except KeyError:
            raise ValueError(
                f"Unsupported pad color: {color}"
            )

    # ------------------------------------------------------------------
    # Core resize helper
    # ------------------------------------------------------------------

    @classmethod
    def resize_to_dimensions(
        cls,
        input_tensor,
        width,
        height,
        scale_method,
        crop="center",
    ):
        """
        Resize an IMAGE or MASK to an exact target rectangle.

        crop="disabled":
            Stretch directly to the requested dimensions.

        crop="center":
            Preserve the source aspect ratio, scale enough to cover the
            requested rectangle, then center-crop to the exact target
            dimensions.

        This helper deliberately owns the final geometry. It does not
        perform any additional multiple-of calculations.
        """

        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        # Already exactly the requested dimensions.
        if (
            source_width == width
            and source_height == height
        ):
            return input_tensor

        if crop not in cls.crop_methods:
            raise ValueError(
                f"Unsupported crop mode: {crop}"
            )

        samples = comfy.utils.common_upscale(
            samples,
            width,
            height,
            scale_method,
            crop,
        )

        return cls.finalize_image_mask_input(
            samples,
            is_type_image,
        )

    # ------------------------------------------------------------------
    # Proportional dimension helpers
    # ------------------------------------------------------------------

    @classmethod
    def dimensions_from_longer(
        cls,
        source_width,
        source_height,
        longer_size,
        multiple_of,
    ):
        """
        Establish the longer dimension first.

        Step 1:
            Round the requested controlling dimension to multiple_of.

        Step 2:
            Calculate the proportional secondary dimension.

        Step 3:
            Round the secondary dimension independently to multiple_of.

        This gives the four dimension-driven modes their two-stage
        multiple-of behavior without requiring a second resize pass.
        """

        longer_size = max(1, int(round(longer_size)))

        if multiple_of > 1:
            longer_size = cls.round_to_multiple(
                longer_size,
                multiple_of,
            )

        if source_width >= source_height:
            width = longer_size

            height = max(
                1,
                round(
                    source_height
                    * width
                    / source_width
                ),
            )

        else:
            height = longer_size

            width = max(
                1,
                round(
                    source_width
                    * height
                    / source_height
                ),
            )

        if multiple_of > 1:
            width, height = (
                cls.round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        return width, height

    @classmethod
    def dimensions_from_shorter(
        cls,
        source_width,
        source_height,
        shorter_size,
        multiple_of,
    ):
        """
        Establish the shorter dimension first.

        The controlling dimension is constrained first, then the
        proportional secondary dimension is constrained independently.
        """

        shorter_size = max(
            1,
            int(round(shorter_size)),
        )

        if multiple_of > 1:
            shorter_size = cls.round_to_multiple(
                shorter_size,
                multiple_of,
            )

        if source_width <= source_height:
            width = shorter_size

            height = max(
                1,
                round(
                    source_height
                    * width
                    / source_width
                ),
            )

        else:
            height = shorter_size

            width = max(
                1,
                round(
                    source_width
                    * height
                    / source_height
                ),
            )

        if multiple_of > 1:
            width, height = (
                cls.round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        return width, height

    @classmethod
    def dimensions_from_width(
        cls,
        source_width,
        source_height,
        target_width,
        multiple_of,
    ):
        """
        Establish width first, then constrain the proportional height.
        """

        target_width = max(
            1,
            int(round(target_width)),
        )

        if multiple_of > 1:
            target_width = cls.round_to_multiple(
                target_width,
                multiple_of,
            )

        target_height = max(
            1,
            round(
                source_height
                * target_width
                / source_width
            ),
        )

        if multiple_of > 1:
            target_height = cls.round_to_multiple(
                target_height,
                multiple_of,
            )

        return target_width, target_height

    @classmethod
    def dimensions_from_height(
        cls,
        source_width,
        source_height,
        target_height,
        multiple_of,
    ):
        """
        Establish height first, then constrain the proportional width.
        """

        target_height = max(
            1,
            int(round(target_height)),
        )

        if multiple_of > 1:
            target_height = cls.round_to_multiple(
                target_height,
                multiple_of,
            )

        target_width = max(
            1,
            round(
                source_width
                * target_height
                / source_height
            ),
        )

        if multiple_of > 1:
            target_width = cls.round_to_multiple(
                target_width,
                multiple_of,
            )

        return target_width, target_height

    # ------------------------------------------------------------------
    # Scale dimensions
    # ------------------------------------------------------------------

    @classmethod
    def scale_dimensions(
        cls,
        input_tensor,
        width,
        height,
        scale_method,
        crop="center",
        multiple_of=0,
    ):
        """
        Resize to explicit dimensions.

        A zero width or height means that dimension is calculated from
        the source aspect ratio.

        Once both dimensions are established, multiple_of constrains
        the final target rectangle.
        """

        if width == 0 and height == 0:
            return input_tensor

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        if width == 0:
            width = max(
                1,
                round(
                    source_width
                    * height
                    / source_height
                ),
            )

        elif height == 0:
            height = max(
                1,
                round(
                    source_height
                    * width
                    / source_width
                ),
            )

        if multiple_of > 1:
            width, height = (
                cls.round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale by multiplier
    # ------------------------------------------------------------------

    @classmethod
    def scale_by(
        cls,
        input_tensor,
        multiplier,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Scale by a multiplier.

        The multiplier establishes the natural target dimensions.
        multiple_of then constrains both resulting dimensions.

        The final crop setting determines whether aspect-ratio mismatch
        caused by the multiple constraint is stretched or center-cropped.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        width = max(
            1,
            round(source_width * multiplier),
        )

        height = max(
            1,
            round(source_height * multiplier),
        )

        if multiple_of > 1:
            width, height = (
                cls.round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale longer dimension
    # ------------------------------------------------------------------

    @classmethod
    def scale_longer_dimension(
        cls,
        input_tensor,
        longer_size,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Scale the longer dimension.

        The longer dimension is constrained first, then the proportional
        secondary dimension is independently constrained.

        The final crop mode determines how the resulting small aspect
        ratio discrepancy is handled.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        width, height = cls.dimensions_from_longer(
            source_width,
            source_height,
            longer_size,
            multiple_of,
        )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale shorter dimension
    # ------------------------------------------------------------------

    @classmethod
    def scale_shorter_dimension(
        cls,
        input_tensor,
        shorter_size,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Scale the shorter dimension.

        The shorter dimension is constrained first, then the proportional
        secondary dimension is independently constrained.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        width, height = cls.dimensions_from_shorter(
            source_width,
            source_height,
            shorter_size,
            multiple_of,
        )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale width
    # ------------------------------------------------------------------

    @classmethod
    def scale_width(
        cls,
        input_tensor,
        width,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Scale to a requested width while preserving aspect ratio.

        Width is constrained first, then the calculated height is
        independently constrained.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        target_width, target_height = (
            cls.dimensions_from_width(
                source_width,
                source_height,
                width,
                multiple_of,
            )
        )

        return cls.resize_to_dimensions(
            input_tensor,
            target_width,
            target_height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale height
    # ------------------------------------------------------------------

    @classmethod
    def scale_height(
        cls,
        input_tensor,
        height,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Scale to a requested height while preserving aspect ratio.

        Height is constrained first, then the calculated width is
        independently constrained.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        target_width, target_height = (
            cls.dimensions_from_height(
                source_width,
                source_height,
                height,
                multiple_of,
            )
        )

        return cls.resize_to_dimensions(
            input_tensor,
            target_width,
            target_height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale total pixels
    # ------------------------------------------------------------------

    @classmethod
    def dimensions_for_total_pixels(
        cls,
        source_width,
        source_height,
        megapixels,
        multiple_of,
    ):
        """
        Calculate dimensions near the requested megapixel target.

        Without multiple_of:
            Use the normal aspect-ratio-preserving calculation.

        With multiple_of:
            Search nearby multiple-compatible width/height pairs and
            select the pair with the smallest total-pixel error.

        Aspect-ratio error is used as the secondary comparison so that
        two equally good area candidates favor the original AR.
        """

        target_pixels = max(
            1,
            int(round(
                megapixels
                * 1024
                * 1024
            )),
        )

        aspect_ratio = (
            source_width
            / source_height
        )

        ideal_width = math.sqrt(
            target_pixels
            * aspect_ratio
        )

        ideal_height = math.sqrt(
            target_pixels
            / aspect_ratio
        )

        if multiple_of <= 1:
            return (
                max(1, round(ideal_width)),
                max(1, round(ideal_height)),
            )

        base_width = cls.round_to_multiple(
            ideal_width,
            multiple_of,
        )

        base_height = cls.round_to_multiple(
            ideal_height,
            multiple_of,
        )

        search_radius = 4

        candidates = []

        for width_offset in range(
            -search_radius,
            search_radius + 1,
        ):
            width = (
                base_width
                + width_offset * multiple_of
            )

            if width < multiple_of:
                continue

            for height_offset in range(
                -search_radius,
                search_radius + 1,
            ):
                height = (
                    base_height
                    + height_offset * multiple_of
                )

                if height < multiple_of:
                    continue

                area = width * height

                area_error = abs(
                    area - target_pixels
                ) / target_pixels

                candidate_aspect = (
                    width / height
                )

                aspect_error = abs(
                    candidate_aspect
                    - aspect_ratio
                ) / aspect_ratio

                candidates.append(
                    (
                        area_error,
                        aspect_error,
                        width,
                        height,
                    )
                )

        if not candidates:
            return (
                base_width,
                base_height,
            )

        candidates.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1],
            )
        )

        _, _, width, height = candidates[0]

        return width, height

    @classmethod
    def scale_total_pixels(
        cls,
        input_tensor,
        megapixels,
        scale_method,
        multiple_of=0,
        crop="center",
    ):
        """
        Resize to approximately the requested megapixel count.

        When multiple_of is active, dimensions are selected directly
        from nearby valid multiple-compatible rectangles instead of
        resizing first and then arbitrarily rounding the result.
        """

        is_type_image = cls.is_image(input_tensor)

        samples = cls.init_image_mask_input(
            input_tensor,
            is_type_image,
        )

        source_width = samples.shape[-1]
        source_height = samples.shape[-2]

        width, height = (
            cls.dimensions_for_total_pixels(
                source_width,
                source_height,
                megapixels,
                multiple_of,
            )
        )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Match size
    # ------------------------------------------------------------------

    @classmethod
    def scale_match_size(
        cls,
        input_tensor,
        match,
        scale_method,
        crop="center",
        multiple_of=0,
    ):
        """
        Resize to the dimensions of another IMAGE or MASK.

        multiple_of constrains the reference dimensions before the
        final resize is performed.
        """

        match_is_image = cls.is_image(match)

        match_samples = cls.init_image_mask_input(
            match,
            match_is_image,
        )

        width = match_samples.shape[-1]
        height = match_samples.shape[-2]

        if multiple_of > 1:
            width, height = (
                cls.round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        return cls.resize_to_dimensions(
            input_tensor,
            width,
            height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # Scale to multiple
    # ------------------------------------------------------------------

    @classmethod
    def scale_to_multiple(
        cls,
        input_tensor,
        multiple,
        scale_method,
        crop="center",
    ):
        """
        Resize so both dimensions are divisible by the specified multiple.

        crop="disabled":
            Resize directly to the independently floored dimensions,
            allowing aspect-ratio distortion.

        crop="center":
            Preserve aspect ratio, scale enough to cover the target
            dimensions, then center-crop to the target rectangle.
        """

        if multiple <= 1:
            return input_tensor

        is_type_image = cls.is_image(input_tensor)

        if is_type_image:
            _, height, width, _ = input_tensor.shape
        else:
            _, height, width = input_tensor.shape

        target_width = cls.floor_to_multiple(
            width,
            multiple,
        )

        target_height = cls.floor_to_multiple(
            height,
            multiple,
        )

        if (
            target_width == width
            and target_height == height
        ):
            return input_tensor

        return cls.resize_to_dimensions(
            input_tensor,
            target_width,
            target_height,
            scale_method,
            crop,
        )

    # ------------------------------------------------------------------
    # From SmartImageResize functionality
    # ------------------------------------------------------------------

    @classmethod
    def legacy_resize(
        cls,
        image,
        width,
        height,
        method,
        multiple_of,
    ):
        """
        Calculate the dimensions and crop/pad geometry for the legacy
        resize modes.

        multiple_of remains part of the geometry calculation.
        """

        oh = image.shape[1]
        ow = image.shape[2]

        x = 0
        y = 0
        x2 = 0
        y2 = 0

        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

        # --------------------------------------------------------------
        # Keep proportion
        # --------------------------------------------------------------

        if method == "keep proportion":

            avg = avg_from_dims(
                width,
                height,
            )

            n, d = ar_parts_from_dims(
                ow,
                oh,
            )

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

            new_width = round(
                ow * ratio
            )

            new_height = round(
                oh * ratio
            )

            x = (
                new_width - width
            ) // 2

            y = (
                new_height - height
            ) // 2

            x2 = x + width
            y2 = y + height

            if x2 > new_width:
                x -= (
                    x2 - new_width
                )

            if x < 0:
                x = 0

            if y2 > new_height:
                y -= (
                    y2 - new_height
                )

            if y < 0:
                y = 0

            width = new_width
            height = new_height

        # --------------------------------------------------------------
        # Pad
        # --------------------------------------------------------------

        elif method == "pad":

            width = (
                width
                if width > 0
                else ow
            )

            height = (
                height
                if height > 0
                else oh
            )

            if multiple_of > 1:
                width, height = (
                    cls.round_dimensions_to_multiple(
                        width,
                        height,
                        multiple_of,
                    )
                )

            ratio = min(
                width / ow,
                height / oh,
            )

            new_width = round(
                ow * ratio
            )

            new_height = round(
                oh * ratio
            )

            pad_left = (
                width - new_width
            ) // 2

            pad_right = (
                width
                - new_width
                - pad_left
            )

            pad_top = (
                height - new_height
            ) // 2

            pad_bottom = (
                height
                - new_height
                - pad_top
            )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        return (
            new_width,
            new_height,
            x,
            y,
            x2,
            y2,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        )

    # ------------------------------------------------------------------
    # Resize one input
    # ------------------------------------------------------------------

    @classmethod
    def resize_input(
        cls,
        input_tensor,
        resize_type,
        scale_method,
        condition="always",
    ):
        """
        Apply the selected resize operation to one IMAGE or MASK.

        This method intentionally knows nothing about whether the input
        came from the image or mask socket. The exact same resize
        configuration can therefore be applied independently to both.
        """

        if input_tensor is None:
            return None

        selected_type = resize_type[
            "resize_type"
        ]

        # --------------------------------------------------------------
        # Source dimensions
        # --------------------------------------------------------------

        source_height = input_tensor.shape[1]
        source_width = input_tensor.shape[2]

        # --------------------------------------------------------------
        # Native-style resize modes
        # --------------------------------------------------------------

        if selected_type == "scale dimensions":

            target_width = resize_type["width"]
            target_height = resize_type["height"]

            outputs = cls.scale_dimensions(
                input_tensor,
                target_width,
                target_height,
                scale_method,
                resize_type.get(
                    "crop",
                    "disabled",
                ),
                resize_type.get(
                    "multiple_of",
                    0,
                ),
            )

        elif selected_type == "scale by multiplier":

            outputs = cls.scale_by(
                input_tensor,
                resize_type["multiplier"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "scale longer dimension":

            outputs = cls.scale_longer_dimension(
                input_tensor,
                resize_type["longer_size"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "scale shorter dimension":

            outputs = cls.scale_shorter_dimension(
                input_tensor,
                resize_type["shorter_size"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "scale width":

            outputs = cls.scale_width(
                input_tensor,
                resize_type["width"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "scale height":

            outputs = cls.scale_height(
                input_tensor,
                resize_type["height"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "scale total pixels":

            outputs = cls.scale_total_pixels(
                input_tensor,
                resize_type["megapixels"],
                scale_method,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
                resize_type.get(
                    "crop",
                    "disabled",
                ),
            )

        elif selected_type == "match size":

            outputs = cls.scale_match_size(
                input_tensor,
                resize_type["match"],
                scale_method,
                resize_type.get(
                    "crop",
                    "center",
                ),
                resize_type.get(
                    "multiple_of",
                    0,
                ),
            )

        elif selected_type == "scale to multiple":

            outputs = cls.scale_to_multiple(
                input_tensor,
                resize_type["multiple"],
                scale_method,
                resize_type.get(
                    "crop",
                    "center",
                ),
            )

        # --------------------------------------------------------------
        # From SmartImageResize modes
        # --------------------------------------------------------------

        elif selected_type in cls.legacy_methods:

            (
                new_width,
                new_height,
                x,
                y,
                x2,
                y2,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ) = cls.legacy_resize(
                input_tensor,
                resize_type["width"],
                resize_type["height"],
                selected_type,
                resize_type.get(
                    "multiple_of",
                    0,
                ),
            )

            # ----------------------------------------------------------
            # Condition
            # ----------------------------------------------------------

            should_resize = (
                condition == "always"
                or (
                    condition == "downscale if bigger"
                    and (
                        source_height > new_height
                        or source_width > new_width
                    )
                )
                or (
                    condition == "upscale if smaller"
                    and (
                        source_height < new_height
                        or source_width < new_width
                    )
                )
                or (
                    condition == "if bigger area"
                    and (
                        source_height * source_width
                        > new_width * new_height
                    )
                )
                or (
                    condition == "if smaller area"
                    and (
                        source_height * source_width
                        < new_width * new_height
                    )
                )
            )

            if should_resize:

                is_type_image = cls.is_image(
                    input_tensor
                )

                samples = cls.init_image_mask_input(
                    input_tensor,
                    is_type_image,
                )

                # ------------------------------------------------------
                # Resize
                # ------------------------------------------------------

                if scale_method == "lanczos":

                    samples = comfy.utils.lanczos(
                        samples,
                        new_width,
                        new_height,
                    )

                else:

                    samples = F.interpolate(
                        samples,
                        size=(
                            new_height,
                            new_width,
                        ),
                        mode=scale_method,
                    )

                # ------------------------------------------------------
                # Pad
                # ------------------------------------------------------

                if selected_type == "pad":

                    if (
                        pad_left
                        or pad_right
                        or pad_top
                        or pad_bottom
                    ):

                        pad_value = cls.pad_color_value(
                            resize_type.get(
                                "color",
                                "black",
                            )
                        )

                        samples = F.pad(
                            samples,
                            (
                                pad_left,
                                pad_right,
                                pad_top,
                                pad_bottom,
                            ),
                            value=pad_value,
                        )

                outputs = cls.finalize_image_mask_input(
                    samples,
                    is_type_image,
                )

            else:

                outputs = input_tensor

        else:

            raise ValueError(
                f"Unsupported resize type: "
                f"{selected_type}"
            )

        # --------------------------------------------------------------
        # Final safety check
        # --------------------------------------------------------------

        if (
            outputs.shape[1] <= 0
            or outputs.shape[2] <= 0
        ):
            raise RuntimeError(
                "ResizeImageMaskAlt produced an invalid "
                f"output size: {outputs.shape}"
            )

        # --------------------------------------------------------------
        # Clamp
        # --------------------------------------------------------------

        return torch.clamp(
            outputs,
            0,
            1,
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @classmethod
    def define_schema(cls):

        resize_options = [

            # ----------------------------------------------------------
            # Native-style resize modes
            # ----------------------------------------------------------

            io.DynamicCombo.Option(
                "scale dimensions",
                [
                    io.Int.Input(
                        "width",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target width. "
                            "0 calculates it from height."
                        ),
                    ),

                    io.Int.Input(
                        "height",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target height. "
                            "0 calculates it from width."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Constrain the final target dimensions "
                            "to the nearest multiple of this value."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle aspect-ratio mismatch. "
                            "'disabled' stretches to fit; "
                            "'center' crops to maintain aspect ratio."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale by multiplier",
                [
                    io.Float.Input(
                        "multiplier",
                        default=1.0,
                        min=0.01,
                        max=8.0,
                        step=0.01,
                        tooltip=(
                            "Scale factor. "
                            "2.0 doubles the dimensions; "
                            "0.5 halves them."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Constrain both resulting dimensions "
                            "to the nearest multiple of this value."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the aspect-ratio mismatch "
                            "created by the multiple constraint. "
                            "'disabled' stretches; "
                            "'center' preserves aspect ratio and "
                            "center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale longer dimension",
                [
                    io.Int.Input(
                        "longer_size",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize the longer edge to this value "
                            "while preserving aspect ratio."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "First constrains the requested longer "
                            "dimension, then constrains the calculated "
                            "secondary dimension."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the small aspect-ratio "
                            "difference created by the multiple "
                            "constraint. 'disabled' stretches; "
                            "'center' preserves aspect ratio and "
                            "center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale shorter dimension",
                [
                    io.Int.Input(
                        "shorter_size",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize the shorter edge to this value "
                            "while preserving aspect ratio."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "First constrains the requested shorter "
                            "dimension, then constrains the calculated "
                            "secondary dimension."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the small aspect-ratio "
                            "difference created by the multiple "
                            "constraint. 'disabled' stretches; "
                            "'center' preserves aspect ratio and "
                            "center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale width",
                [
                    io.Int.Input(
                        "width",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target width. "
                            "Height is calculated automatically."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "First constrains the requested width, "
                            "then constrains the calculated height."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the small aspect-ratio "
                            "difference created by the multiple "
                            "constraint. 'disabled' stretches; "
                            "'center' preserves aspect ratio and "
                            "center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale height",
                [
                    io.Int.Input(
                        "height",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target height. "
                            "Width is calculated automatically."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "First constrains the requested height, "
                            "then constrains the calculated width."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the small aspect-ratio "
                            "difference created by the multiple "
                            "constraint. 'disabled' stretches; "
                            "'center' preserves aspect ratio and "
                            "center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale total pixels",
                [
                    io.Float.Input(
                        "megapixels",
                        default=1.0,
                        min=0.01,
                        max=16.0,
                        step=0.01,
                        tooltip=(
                            "Target total megapixels. "
                            "1.0 is approximately 1,048,576 pixels."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Find dimensions near the requested "
                            "megapixel target that are both divisible "
                            "by this value."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle the aspect-ratio mismatch "
                            "created by the constrained dimensions. "
                            "'disabled' stretches; 'center' preserves "
                            "aspect ratio and center-crops."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "match size",
                [
                    io.MultiType.Input(
                        "match",
                        [
                            io.Image,
                            io.Mask,
                        ],
                        tooltip=(
                            "Resize to match the dimensions "
                            "of this image or mask."
                        ),
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Constrain the reference dimensions "
                            "to the nearest multiple of this value."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle aspect-ratio mismatch "
                            "when matching the reference size."
                        ),
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "scale to multiple",
                [
                    io.Int.Input(
                        "multiple",
                        default=8,
                        min=1,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize while preserving aspect ratio "
                            "so the final width and height are "
                            "divisible by this value."
                        ),
                    ),

                    io.Combo.Input(
                        "crop",
                        options=cls.crop_methods,
                        default="center",
                        tooltip=(
                            "How to handle aspect-ratio mismatch "
                            "when matching the reference size."
                        ),
                    ),
                ],
            ),

            # ----------------------------------------------------------
            # From SmartImageResize modes
            # ----------------------------------------------------------

            io.DynamicCombo.Option(
                "keep proportion",
                [
                    io.Int.Input(
                        "width",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                    ),

                    io.Int.Input(
                        "height",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Constrain the target geometry "
                            "to this multiple."
                        ),
                        advanced=True,
                    ),
                ],
            ),

            io.DynamicCombo.Option(
                "pad",
                [
                    io.Int.Input(
                        "width",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                    ),

                    io.Int.Input(
                        "height",
                        default=512,
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                    ),

                    io.Int.Input(
                        "multiple_of",
                        default=0,
                        min=0,
                        max=512,
                        step=1,
                        tooltip=(
                            "Constrain the padded dimensions "
                            "to this multiple."
                        ),
                        advanced=True,
                    ),

                    io.Combo.Input(
                        "color",
                        options=cls.pad_colors,
                        default="black",
                        tooltip=(
                            "Color used for the padded area."
                        ),
                    ),
                ],
            ),
        ]

        return io.Schema(
            node_id="ResizeImageMaskAlt",
            display_name="Resize Image/Mask Alt ◯",
            description=("Resize an image, a mask, or both using multiple methods while optionally preserving aspect ratio and enforcing dimension multiples."),
            category="essentials/image manipulation",
            search_aliases=[
                "resize",
                "resize image",
                "resize mask",
                "smart resize",
                "image resize",
                "mask resize",
                "scale",
                "dimensions",
                "megapixels",
                "aspect ratio",
                "crop",
                "pad",
            ],

            inputs=[
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Optional image to resize."
                    ),
                ),

                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional mask to resize."
                    ),
                ),

                io.DynamicCombo.Input(
                    "resize_type",
                    options=resize_options,
                    tooltip=(
                        "Select the resize operation. "
                        "The same configuration is applied "
                        "independently to the image and mask inputs."
                    ),
                ),

                io.Combo.Input(
                    "scale_method",
                    options=cls.scale_methods,
                    default="area",
                    tooltip=(
                        "Interpolation algorithm."
                    ),
                ),

                io.Combo.Input(
                    "condition",
                    options=cls.conditions,
                    default="always",
                    tooltip=(
                        "Optional condition controlling whether resize operation is performed."
                    ),
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="resized_image",
                ),

                io.Mask.Output(
                    display_name="resized_mask",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        image=None,
        mask=None,
        resize_type=None,
        scale_method="area",
        condition="always",
    ) -> io.NodeOutput:

        # --------------------------------------------------------------
        # Require at least one input
        # --------------------------------------------------------------

        if image is None and mask is None:
            raise ValueError(
                "ResizeImageMaskAlt requires at least one input: "
                "image or mask."
            )

        # --------------------------------------------------------------
        # Apply the exact same resize configuration independently
        # to each supplied input.
        #
        # Each input calculates its geometry from its own dimensions.
        # --------------------------------------------------------------

        resized_image = None
        resized_mask = None

        if image is not None:

            resized_image = cls.resize_input(
                image,
                resize_type,
                scale_method,
                condition,
            )

        if mask is not None:

            resized_mask = cls.resize_input(
                mask,
                resize_type,
                scale_method,
                condition,
            )

        # --------------------------------------------------------------
        # Final output
        #
        # NodeOutput values correspond directly to:
        #
        #   1. resized_image
        #   2. resized_mask
        #
        # None is intentionally returned for an unused input.
        # --------------------------------------------------------------

        return io.NodeOutput(
            resized_image,
            resized_mask,
        )


class SmartImageResizeAlt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SmartImageResizeAlt",
            display_name="Smart Image Resize Alt ◯",
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
