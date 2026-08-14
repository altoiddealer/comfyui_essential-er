import math
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.utils
from nodes import MAX_RESOLUTION
from comfy_api.latest import io

from .utils_image import round_to_multiple, round_dimensions_to_multiple, floor_to_multiple, is_image_tensor, image_mask_to_nchw, nchw_to_image_mask, pad_color_value, image_mask_dimensions, \
                         avg_from_dims, ar_parts_from_dims, dims_from_ar, ar_parts_from_str
from .utils_user_overrides import UserOverrides
overrides = UserOverrides()

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

    default_aspect_ratios = [
        "1:1 (Square)",
        "2:3 (Portrait Photo)",
        "3:2 (Photo)",
        "3:4 (Portrait Standard)",
        "4:3 (Standard)",
        "9:16 (Portrait Widescreen)",
        "16:9 (Widescreen)",
        "9:21 (Portrait Ultrawide)",
        "21:9 (Ultrawide)",
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
    # Aspect Ratio helper
    # ------------------------------------------------------------------

    @classmethod
    def get_aspect_ratio_options(cls):

        options = ["Source (Keep Aspect Ratio)"]

        options.extend(
            overrides.get("nodes_image", "resize_image_mask_alt", "smart_resize", "aspect_ratios",
                          default=cls.default_aspect_ratios)
                          )

        return options

    # ------------------------------------------------------------------
    # Condition check helper
    # ------------------------------------------------------------------

    @staticmethod
    def should_resize(
        source_width,
        source_height,
        target_width,
        target_height,
        condition,
    ):
        """
        Determine whether a resize operation should be performed.
        """

        if condition == "always":
            return True

        if condition == "downscale if bigger":

            return (
                source_width > target_width
                or source_height > target_height
            )

        if condition == "upscale if smaller":

            return (
                source_width < target_width
                or source_height < target_height
            )

        source_area = (
            source_width
            * source_height
        )

        target_area = (
            target_width
            * target_height
        )

        if condition == "if bigger area":
            return source_area > target_area

        if condition == "if smaller area":
            return source_area < target_area

        raise ValueError(
            f"Unsupported resize condition: "
            f"{condition}"
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

        is_type_image = is_image_tensor(input_tensor)

        samples = image_mask_to_nchw(input_tensor)

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

        return nchw_to_image_mask(
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
            longer_size = round_to_multiple(
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
                round_dimensions_to_multiple(
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
            shorter_size = round_to_multiple(
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
                round_dimensions_to_multiple(
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
            target_width = round_to_multiple(
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
            target_height = round_to_multiple(
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
            target_height = round_to_multiple(
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
            target_width = round_to_multiple(
                target_width,
                multiple_of,
            )

        return target_width, target_height


    # ------------------------------------------------------------------
    # Smart Resize
    # ------------------------------------------------------------------

    @classmethod
    def smart_resize_dimensions(
        cls,
        image,
        aspect_ratio,
        megapixels,
        megapixel_priority,
        multiple_of=0,
    ):
        """
        Calculate the final target dimensions for Smart Resize.

        When megapixels > 0:
            Target total pixel count is determined by megapixels.
            The selected aspect ratio determines the target geometry.
            When multiple_of is active, megapixel_priority controls
            the tradeoff between total-pixel accuracy and aspect-ratio
            accuracy.

        When megapixels <= 0:
            Retain image pixel count as a base
        """

        oh = image.shape[1]
        ow = image.shape[2]

        # Determine target aspect ratio.
        if (
            not aspect_ratio
            or aspect_ratio == "Source (Keep Aspect Ratio)"
        ):
            n, d = ar_parts_from_dims(ow, oh)

        else:
            n, d = ar_parts_from_str(aspect_ratio)

        # ----------------------------------------------------------
        # Megapixels strategy
        # ----------------------------------------------------------

        if megapixels > 0:

            target_pixels = (
                megapixels
                * 1024
                * 1024
            )

            target_width, target_height = (
                cls.dimensions_for_target_pixels(
                    target_pixels,
                    n / d,
                    megapixel_priority,
                    multiple_of,
                )
            )

            return (
                target_width,
                target_height,
            )

        # ----------------------------------------------------------
        # Width/height fallback strategy
        # ----------------------------------------------------------

        width = ow
        height = oh

        target_resolution = avg_from_dims(
            width,
            height,
        )

        target_width, target_height = (
            dims_from_ar(
                target_resolution,
                n,
                d,
                multiple_of,
            )
        )

        return (
            target_width,
            target_height,
        )

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
        multiple_of=0,
        crop="center",
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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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
                round_dimensions_to_multiple(
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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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
                round_dimensions_to_multiple(
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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

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
    def dimensions_for_target_pixels(
        cls,
        target_pixels,
        aspect_ratio,
        megapixel_priority=1.0,
        multiple_of=0,
    ):
        """
        Calculate dimensions near the requested pixel count while
        balancing total-pixel accuracy against aspect-ratio accuracy.

        megapixel_priority:
            1.0 = prioritize total-pixel accuracy.
            0.0 = prioritize aspect-ratio accuracy.
            Values between 0.0 and 1.0 blend the two priorities.

        When multiple_of is active, only dimension pairs that satisfy
        the constraint are considered.
        """

        target_pixels = max(
            1,
            int(round(target_pixels)),
        )

        aspect_ratio = max(
            1e-12,
            float(aspect_ratio),
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

        # Keep the setting safely within its documented range.
        megapixel_priority = max(
            0.0,
            min(1.0, float(megapixel_priority)),
        )

        base_width = round_to_multiple(
            ideal_width,
            multiple_of,
        )

        base_height = round_to_multiple(
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

                # Blend megapixel accuracy and aspect-ratio accuracy.
                score = (
                    megapixel_priority * area_error
                    + (1.0 - megapixel_priority) * aspect_error
                )

                candidates.append(
                    (
                        score,
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
                candidate[2],
            )
        )

        _, _, _, width, height = candidates[0]

        return width, height

    @classmethod
    def dimensions_for_total_pixels(
        cls,
        source_width,
        source_height,
        megapixels,
        megapixel_priority=1.0,
        multiple_of=0,
    ):
        """
        Calculate dimensions near the requested megapixel target while
        preserving the source aspect ratio.

        megapixel_priority:
            1.0 = prioritize total-pixel accuracy.
            0.0 = prioritize aspect-ratio accuracy.
            Values between 0.0 and 1.0 blend the two priorities.

        When multiple_of is active, only dimension pairs that satisfy
        the constraint are considered.
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

        return cls.dimensions_for_target_pixels(
            target_pixels,
            aspect_ratio,
            megapixel_priority,
            multiple_of,
        )

    @classmethod
    def scale_total_pixels(
        cls,
        input_tensor,
        megapixels,
        megapixel_priority=1.0,
        scale_method="area",
        multiple_of=0,
        crop="center",
    ):
        """
        Resize to approximately the requested megapixel count.

        megapixel_priority controls the tradeoff between hitting the
        requested megapixel count and preserving the source aspect ratio.

        When multiple_of is active, dimensions are selected directly
        from nearby valid multiple-compatible rectangles.
        """

        source_width, source_height = (
            image_mask_dimensions(input_tensor)
        )

        width, height = (
            cls.dimensions_for_total_pixels(
                source_width,
                source_height,
                megapixels,
                megapixel_priority,
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
        multiple_of=0,
        crop="center",
    ):
        """
        Resize to the dimensions of another IMAGE or MASK.

        multiple_of constrains the reference dimensions before the
        final resize is performed.
        """

        width, height = (
            image_mask_dimensions(match)
        )

        if multiple_of > 1:
            width, height = (
                round_dimensions_to_multiple(
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

        is_type_image = is_image_tensor(input_tensor)

        if is_type_image:
            _, height, width, _ = input_tensor.shape
        else:
            _, height, width = input_tensor.shape

        target_width = floor_to_multiple(
            width,
            multiple,
        )

        target_height = floor_to_multiple(
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
    # Pad
    # ------------------------------------------------------------------

    @classmethod
    def pad_dimensions(
        cls,
        input_tensor,
        width,
        height,
        multiple_of=0,
    ):
        """
        Calculate the resize and padding geometry for the pad mode.

        width and height define the final padded rectangle.

        The source is scaled proportionally so that it fits entirely within
        that rectangle, then centered within the remaining space.

        multiple_of constrains the final padded dimensions.
        """

        source_height = input_tensor.shape[1]
        source_width = input_tensor.shape[2]

        width = (
            width
            if width > 0
            else source_width
        )

        height = (
            height
            if height > 0
            else source_height
        )

        if multiple_of > 1:
            width, height = (
                round_dimensions_to_multiple(
                    width,
                    height,
                    multiple_of,
                )
            )

        ratio = min(
            width / source_width,
            height / source_height,
        )

        resized_width = max(
            1,
            round(
                source_width * ratio
            ),
        )

        resized_height = max(
            1,
            round(
                source_height * ratio
            ),
        )

        pad_left = (
            width - resized_width
        ) // 2

        pad_right = (
            width
            - resized_width
            - pad_left
        )

        pad_top = (
            height - resized_height
        ) // 2

        pad_bottom = (
            height
            - resized_height
            - pad_top
        )

        return (
            width,
            height,
            resized_width,
            resized_height,
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

        The condition is evaluated against the final target dimensions
        before the resize operation is performed.
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
        # Smart Resize
        # --------------------------------------------------------------

        if selected_type == "smart_resize":

            target_width, target_height = (
                cls.smart_resize_dimensions(
                    input_tensor,
                    resize_type["aspect_ratio"],
                    resize_type["megapixels"],
                    resize_type["megapixel_priority"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.resize_to_dimensions(
                    input_tensor,
                    target_width,
                    target_height,
                    scale_method,
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale dimensions
        # --------------------------------------------------------------

        elif selected_type == "scale dimensions":

            width = resize_type["width"]
            height = resize_type["height"]
            multiple_of = resize_type["multiple_of"]

            # A zero width/height means that dimension is derived from
            # the source aspect ratio.
            if width == 0 and height == 0:

                target_width = source_width
                target_height = source_height

            else:

                if width == 0:

                    target_width = max(
                        1,
                        round(
                            source_width
                            * height
                            / source_height
                        ),
                    )

                    target_height = height

                elif height == 0:

                    target_width = width

                    target_height = max(
                        1,
                        round(
                            source_height
                            * width
                            / source_width
                        ),
                    )

                else:

                    target_width = width
                    target_height = height

                if multiple_of > 1:

                    target_width, target_height = (
                        round_dimensions_to_multiple(
                            target_width,
                            target_height,
                            multiple_of,
                        )
                    )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_dimensions(
                    input_tensor,
                    width,
                    height,
                    scale_method,
                    multiple_of,
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale by multiplier
        # --------------------------------------------------------------

        elif selected_type == "scale by multiplier":

            multiplier = resize_type["multiplier"]
            multiple_of = resize_type["multiple_of"]

            target_width = max(
                1,
                round(
                    source_width * multiplier
                ),
            )

            target_height = max(
                1,
                round(
                    source_height * multiplier
                ),
            )

            if multiple_of > 1:

                target_width, target_height = (
                    round_dimensions_to_multiple(
                        target_width,
                        target_height,
                        multiple_of,
                    )
                )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_by(
                    input_tensor,
                    multiplier,
                    scale_method,
                    multiple_of,
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale longer dimension
        # --------------------------------------------------------------

        elif selected_type == "scale longer dimension":

            target_width, target_height = (
                cls.dimensions_from_longer(
                    source_width,
                    source_height,
                    resize_type["longer_size"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_longer_dimension(
                    input_tensor,
                    resize_type["longer_size"],
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale shorter dimension
        # --------------------------------------------------------------

        elif selected_type == "scale shorter dimension":

            target_width, target_height = (
                cls.dimensions_from_shorter(
                    source_width,
                    source_height,
                    resize_type["shorter_size"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_shorter_dimension(
                    input_tensor,
                    resize_type["shorter_size"],
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale width
        # --------------------------------------------------------------

        elif selected_type == "scale width":

            target_width, target_height = (
                cls.dimensions_from_width(
                    source_width,
                    source_height,
                    resize_type["width"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_width(
                    input_tensor,
                    resize_type["width"],
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale height
        # --------------------------------------------------------------

        elif selected_type == "scale height":

            target_width, target_height = (
                cls.dimensions_from_height(
                    source_width,
                    source_height,
                    resize_type["height"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_height(
                    input_tensor,
                    resize_type["height"],
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale total pixels
        # --------------------------------------------------------------

        elif selected_type == "scale total pixels":

            target_width, target_height = (
                cls.dimensions_for_total_pixels(
                    source_width,
                    source_height,
                    resize_type["megapixels"],
                    resize_type["megapixel_priority"],
                    resize_type["multiple_of"],
                )
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_total_pixels(
                    input_tensor,
                    resize_type["megapixels"],
                    resize_type["megapixel_priority"],
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Match size
        # --------------------------------------------------------------

        elif selected_type == "match size":

            match = resize_type["match"]

            target_width, target_height = (
                image_mask_dimensions(match)
            )

            if resize_type["multiple_of"] > 1:

                target_width, target_height = (
                    round_dimensions_to_multiple(
                        target_width,
                        target_height,
                        resize_type["multiple_of"],
                    )
                )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_match_size(
                    input_tensor,
                    match,
                    scale_method,
                    resize_type["multiple_of"],
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Scale to multiple
        # --------------------------------------------------------------

        elif selected_type == "scale to multiple":

            multiple = resize_type["multiple"]

            if multiple <= 1:

                target_width = source_width
                target_height = source_height

            else:

                target_width = floor_to_multiple(
                    source_width,
                    multiple,
                )

                target_height = floor_to_multiple(
                    source_height,
                    multiple,
                )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.scale_to_multiple(
                    input_tensor,
                    multiple,
                    scale_method,
                    resize_type["crop"],
                )

            else:

                outputs = input_tensor

        # --------------------------------------------------------------
        # Pad
        # --------------------------------------------------------------

        elif selected_type == "pad":

            (
                target_width,
                target_height,
                resized_width,
                resized_height,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ) = cls.pad_dimensions(
                input_tensor,
                resize_type["width"],
                resize_type["height"],
                resize_type["multiple_of"],
            )

            if cls.should_resize(
                source_width,
                source_height,
                target_width,
                target_height,
                condition,
            ):

                outputs = cls.resize_to_dimensions(
                    input_tensor,
                    resized_width,
                    resized_height,
                    scale_method,
                    "disabled",
                )

                is_type_image = is_image_tensor(outputs)

                samples = image_mask_to_nchw(outputs)

                if (
                    pad_left
                    or pad_right
                    or pad_top
                    or pad_bottom
                ):

                    pad_value = pad_color_value(resize_type["color"])

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

                outputs = nchw_to_image_mask(
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
    def get_option(cls, name: str):

        if name == "multiple_of":
            return io.Int.Input(
                "multiple_of",
                default=overrides.get("nodes_image", "shared", "resolution", default=0),
                min=0,
                max=512,
                step=1,
                tooltip=("Constrain the dimensions to the nearest multiple of this value."),
                advanced=True,
            )

        elif name == "crop":
            return io.Combo.Input(
                "crop",
                options=cls.crop_methods,
                default="center",
                tooltip=(
                    "How to handle aspect-ratio mismatch. "
                    "'disabled' stretches to fit; "
                    "'center' crops to maintain aspect ratio."
                ),
            )

        elif name == "megapixels":
            return io.Float.Input(
                "megapixels",
                default=overrides.get("nodes_image", "shared", "megapixels", default=1.0),
                min=0.0,
                max=16.0,
                step=0.01,
                tooltip=(
                    "Target total megapixels. "
                    "1.0 is approximately 1,048,576 pixels (≈ 1024×1024). "
                    "Aspect ratio is preserved by default. "
                    "Constraining the dimensions via multiple_of "
                    "requires a tradeoff between megapixel and "
                    "aspect-ratio precision."
                ),
            )

        elif name == "megapixel_priority":
            return io.Float.Input(
                "megapixel_priority",
                default=overrides.get("nodes_image", "shared", "megapixel_priority", default=1.0),
                min=0.0,
                max=1.0,
                step=0.1,
                tooltip=(
                    "While multiple_of > 0: "
                    "Controls the tradeoff between hitting the target megapixel count / aspect ratio. "
                    "1.0 prioritizes megapixel precision (default behavior); "
                    "0.0 prioritizes aspect-ratio precision."
                ),
                advanced=True,
            )

        return None

    @classmethod
    def define_schema(cls):

        resize_options = [

            io.DynamicCombo.Option(
                "smart_resize",
                [
                    io.Combo.Input(
                        "aspect_ratio",
                        options=(cls.get_aspect_ratio_options()),
                        default=("Source (Keep Aspect Ratio)"),
                        tooltip=(
                            "Target aspect ratio. "
                            "The source is scaled proportionally while constrained to precision of 'multiple_of'."
                        ),
                    ),

                    cls.get_option("megapixels"),

                    cls.get_option("megapixel_priority"),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            # ----------------------------------------------------------
            # Native-style resize modes
            # ----------------------------------------------------------

            io.DynamicCombo.Option(
                "scale dimensions",
                [
                    io.Int.Input(
                        "width",
                        default=overrides.get("nodes_image", "shared", "width", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target width. Set to 0 to auto-calculate from height while preserving aspect ratio."
                        ),
                    ),

                    io.Int.Input(
                        "height",
                        default=overrides.get("nodes_image", "shared", "height", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target height. Set to 0 to auto-calculate from width while preserving aspect ratio."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale by multiplier",
                [
                    io.Float.Input(
                        "multiplier",
                        default=overrides.get("nodes_image", "shared", "multiplier", default=1.0),
                        min=0.01,
                        max=8.0,
                        step=0.01,
                        tooltip=(
                            "Scale factor. "
                            "2.0 doubles the dimensions; "
                            "0.5 halves them."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale longer dimension",
                [
                    io.Int.Input(
                        "longer_size",
                        default=overrides.get("nodes_image", "resize_image_mask_alt", "scale_x_dimension", "longer_size", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize the longer edge to this value while preserving aspect ratio."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale shorter dimension",
                [
                    io.Int.Input(
                        "shorter_size",
                        default=overrides.get("nodes_image", "resize_image_mask_alt", "scale_x_dimension", "shorter_size", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize the shorter edge to this value "
                            "while preserving aspect ratio."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale width",
                [
                    io.Int.Input(
                        "width",
                        default=overrides.get("nodes_image", "shared", "width", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target width. Height is calculated automatically."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale height",
                [
                    io.Int.Input(
                        "height",
                        default=overrides.get("nodes_image", "shared", "height", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Target height. Width is calculated automatically."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale total pixels",
                [
                    cls.get_option("megapixels"),

                    cls.get_option("megapixel_priority"),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
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
                            "Resize to match the dimensions of this image or mask."
                        ),
                    ),

                    cls.get_option("multiple_of"),

                    cls.get_option("crop"),
                ],
            ),

            io.DynamicCombo.Option(
                "scale to multiple",
                [
                    io.Int.Input(
                        "multiple",
                        default=overrides.get("nodes_image", "resize_image_mask_alt", "scale_to_multiple", "multiple", default=8),
                        min=1,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Resize while preserving aspect ratio "
                            "so the final width and height are "
                            "divisible by this value."
                        ),
                    ),

                    cls.get_option("crop"),
                ],
            ),

            # ----------------------------------------------------------
            # Pad
            # ----------------------------------------------------------

            io.DynamicCombo.Option(
                "pad",
                [
                    io.Int.Input(
                        "width",
                        default=overrides.get("nodes_image", "shared", "width", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Final padded width. 0 uses the source width."
                        )
                    ),

                    io.Int.Input(
                        "height",
                        default=overrides.get("nodes_image", "shared", "height", default=512),
                        min=0,
                        max=MAX_RESOLUTION,
                        step=1,
                        tooltip=(
                            "Final padded height. 0 uses the source height."
                        ),
                    ),

                    io.Combo.Input(
                        "color",
                        options=cls.pad_colors,
                        default="black",
                        tooltip=(
                            "Color used for the padded area."
                        ),
                    ),

                    cls.get_option("multiple_of"),
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
            display_name="Smart Image Resize Alt ◯ [Deprecated]",
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
