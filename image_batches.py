from comfy_api.latest import io

import torch


class MergeImageBatchList(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MergeImageBatchList",
            display_name="Merge Image Batch List",
            description="Merge a list of image batches together with configurable overlap and blending.",
            category="image/batch",
            search_aliases=[
                "merge images",
                "merge image batches",
                "combine image batches",
                "concat image batches",
                "blend image batches",
                "image batch merge",
            ],

            # V3 equivalent of INPUT_IS_LIST = True.
            is_input_list=True,

            inputs=[
                io.Image.Input(
                    "image_batches_list",
                    tooltip="Image batches to merge together.",
                ),

                io.Int.Input(
                    "overlap",
                    default=13,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Number of images to overlap between consecutive batches.",
                ),

                io.Combo.Input(
                    "overlap_side",
                    options=[
                        "source",
                        "new_images",
                    ],
                    default="source",
                    tooltip="Determines which batch supplies the first side of the overlap.",
                ),

                io.Combo.Input(
                    "overlap_mode",
                    options=[
                        "cut",
                        "linear_blend",
                        "ease_in_out",
                        "filmic_crossfade",
                        "perceptual_crossfade",
                    ],
                    default="linear_blend",
                    tooltip="How overlapping images are combined.",
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="images",
                ),
            ],
        )

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

        overlap = min(
            overlap,
            len(source_images),
            len(new_images),
        )

        if overlap <= 0:
            return torch.cat(
                (source_images, new_images),
                dim=0,
            )

        prefix = source_images[:-overlap]

        if overlap_side == "source":
            blend_src = source_images[-overlap:]
            blend_dst = new_images[:overlap]
        else:
            blend_src = new_images[:overlap]
            blend_dst = source_images[-overlap:]

        suffix = new_images[overlap:]

        if overlap_mode == "linear_blend":
            alpha = torch.linspace(
                0,
                1,
                overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1].view(-1, 1, 1, 1)

            blended = (1 - alpha) * blend_src + alpha * blend_dst

            return torch.cat(
                (prefix, blended, suffix),
                dim=0,
            )

        elif overlap_mode == "ease_in_out":
            t = torch.linspace(
                0,
                1,
                overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1]

            eased = (
                3 * t * t - 2 * t * t * t
            ).view(-1, 1, 1, 1)

            blended = (
                (1 - eased) * blend_src
                + eased * blend_dst
            )

            return torch.cat(
                (prefix, blended, suffix),
                dim=0,
            )

        elif overlap_mode == "filmic_crossfade":
            gamma = 2.2

            alpha = torch.linspace(
                0,
                1,
                overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1].view(-1, 1, 1, 1)

            src = torch.pow(blend_src, gamma)
            dst = torch.pow(blend_dst, gamma)

            blended = (1 - alpha) * src + alpha * dst
            blended = torch.pow(blended, 1.0 / gamma)

            return torch.cat(
                (prefix, blended, suffix),
                dim=0,
            )

        elif overlap_mode == "perceptual_crossfade":
            import kornia

            alpha = torch.linspace(
                0,
                1,
                overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1].view(-1, 1, 1, 1)

            src = blend_src.movedim(-1, 1)
            dst = blend_dst.movedim(-1, 1)

            lab_src = kornia.color.rgb_to_lab(src)
            lab_dst = kornia.color.rgb_to_lab(dst)

            blended = (1 - alpha) * lab_src + alpha * lab_dst
            blended = kornia.color.lab_to_rgb(blended)
            blended = blended.movedim(1, -1)

            return torch.cat(
                (prefix, blended, suffix),
                dim=0,
            )

        elif overlap_mode == "cut":
            if overlap_side == "new_images":
                return torch.cat(
                    (
                        source_images,
                        new_images[overlap:],
                    ),
                    dim=0,
                )

            return torch.cat(
                (
                    source_images[:-overlap],
                    new_images,
                ),
                dim=0,
            )

        raise ValueError(
            f"Unknown overlap mode: {overlap_mode}"
        )

    @classmethod
    def execute(
        cls,
        image_batches_list,
        overlap,
        overlap_side,
        overlap_mode,
    ):
        # Because is_input_list=True, ALL inputs arrive as lists.
        #
        # images:
        #     [batch1, batch2, batch3, ...]
        #
        # overlap:
        #     [13]
        #
        # overlap_side:
        #     ["source"]
        #
        # overlap_mode:
        #     ["linear_blend"]

        if isinstance(overlap, list):
            overlap = overlap[0]

        if isinstance(overlap_side, list):
            overlap_side = overlap_side[0]

        if isinstance(overlap_mode, list):
            overlap_mode = overlap_mode[0]

        if not image_batches_list:
            raise ValueError(
                "No image batches supplied"
            )

        if len(image_batches_list) == 1:
            return io.NodeOutput(image_batches_list[0])

        merged = image_batches_list[0]

        for batch in image_batches_list[1:]:
            merged = cls.merge_batches(
                merged,
                batch,
                overlap,
                overlap_side,
                overlap_mode,
            )

        return io.NodeOutput(merged)
