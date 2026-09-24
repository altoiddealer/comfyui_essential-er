from comfy_api.latest import io

import torch
import torch.nn.functional as F

from .utils_user_overrides import UserOverrides
overrides = UserOverrides()

class MergeImageBatchList(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MergeImageBatchList",
            display_name="Merge Image Batch List Alt ◯",
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
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap", default=13),
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Number of images to overlap between consecutive batches.",
                ),

                io.Combo.Input(
                    "overlap_side",
                    options=[
                        "previous",
                        "next",
                    ],
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap_side", default="previous"),
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
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap_mode", default="linear_blend"),
                    tooltip=(
                        "How overlapping images are combined:\n"
                        "• cut: Hard transition; one batch replaces the other.\n"
                        "• linear_blend: Constant-rate crossfade.\n"
                        "• ease_in_out: Smooth crossfade with gentle transitions at both ends.\n"
                        "• filmic_crossfade: Gamma-adjusted crossfade for more natural brightness.\n"
                        "• perceptual_crossfade: Lab color-space crossfade for more perceptually natural color transitions."
                    ),
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
        previous_images,
        next_images,
        overlap,
        overlap_side,
        overlap_mode,
    ):
        if previous_images.shape[1:3] != next_images.shape[1:3]:
            raise ValueError(
                f"Previous and next images must have same shape: "
                f"{previous_images.shape[1:3]} vs {next_images.shape[1:3]}"
            )

        overlap = min(
            overlap,
            len(previous_images),
            len(next_images),
        )

        if overlap <= 0:
            return torch.cat(
                (previous_images, next_images),
                dim=0,
            )

        prefix = previous_images[:-overlap]

        if overlap_side == "previous":
            blend_src = previous_images[-overlap:]
            blend_dst = next_images[:overlap]
        else:
            blend_src = next_images[:overlap]
            blend_dst = previous_images[-overlap:]

        suffix = next_images[overlap:]

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
            if overlap_side == "next_images":
                return torch.cat(
                    (
                        previous_images,
                        next_images[overlap:],
                    ),
                    dim=0,
                )

            return torch.cat(
                (
                    previous_images[:-overlap],
                    next_images,
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

class MergeImageBatchAndAudioList(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MergeImageBatchAndAudioList",
            display_name="Merge Image Batches and Audio Lists Alt ◯",
            description=(
                "Merges corresponding lists of image batches and audio "
                "with synchronized frame/sample overlap and configurable blending."
            ),
            category="image/batch",
            search_aliases=[
                "merge images audio",
                "merge image audio",
                "merge image batches audio",
                "combine image batches audio",
                "concat image batches audio",
                "blend image batches audio",
                "video batch merge",
            ],

            is_input_list=True,

            inputs=[
                io.Image.Input(
                    "image_batches_list",
                    tooltip="Image batches to merge together.",
                ),

                io.Audio.Input(
                    "audio_list",
                    tooltip=(
                        "Audio corresponding to each image batch. "
                        "The indexes must match the image batch list."
                    ),
                ),

                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.001,
                    max=240.0,
                    step=0.01,
                    tooltip=(
                        "Frame rate used to convert the image overlap "
                        "from frames into an audio overlap."
                    ),
                ),

                io.Int.Input(
                    "overlap",
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap", default=13),
                    min=1,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Number of images/frames to overlap between "
                        "consecutive batches."
                    ),
                ),

                io.Combo.Input(
                    "overlap_side",
                    options=[
                        "previous",
                        "next",
                    ],
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap_side", default="previous"),
                    tooltip=(
                        "Determines which batch supplies the first side "
                        "of the overlap."
                    ),
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
                    default=overrides.get("nodes_batching", "merge_image_batch_list", "overlap_mode", default="linear_blend"),
                    tooltip=(
                        "How overlapping images and audio are combined:\n"
                        "• cut: Hard transition; one batch replaces the other.\n"
                        "• linear_blend: Constant-rate crossfade.\n"
                        "• ease_in_out: Smooth crossfade with gentle transitions at both ends.\n"
                        "• filmic_crossfade: Gamma-adjusted image crossfade with a smooth audio fade.\n"
                        "• perceptual_crossfade: Lab color-space image crossfade with equal-power audio fade."
                    ),
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="images",
                ),

                io.Audio.Output(
                    display_name="audio",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # IMAGE MERGING
    # ------------------------------------------------------------------

    @staticmethod
    def merge_image_batches(
        previous_images,
        next_images,
        overlap,
        overlap_side,
        overlap_mode,
    ):
        if previous_images.shape[1:3] != next_images.shape[1:3]:
            raise ValueError(
                f"Previous and next images must have same shape: "
                f"{previous_images.shape[1:3]} vs {next_images.shape[1:3]}"
            )

        overlap = min(
            overlap,
            len(previous_images),
            len(next_images),
        )

        if overlap <= 0:
            return torch.cat(
                (previous_images, next_images),
                dim=0,
            )

        prefix = previous_images[:-overlap]

        if overlap_side == "previous":
            blend_src = previous_images[-overlap:]
            blend_dst = next_images[:overlap]
        else:
            blend_src = next_images[:overlap]
            blend_dst = previous_images[-overlap:]

        suffix = next_images[overlap:]

        if overlap_mode == "linear_blend":
            alpha = torch.linspace(
                0,
                1,
                overlap + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1].view(-1, 1, 1, 1)

            blended = (
                (1 - alpha) * blend_src
                + alpha * blend_dst
            )

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
                3 * t * t
                - 2 * t * t * t
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

            src = torch.pow(
                torch.clamp(blend_src, min=0),
                gamma,
            )

            dst = torch.pow(
                torch.clamp(blend_dst, min=0),
                gamma,
            )

            blended = (
                (1 - alpha) * src
                + alpha * dst
            )

            blended = torch.pow(
                torch.clamp(blended, min=0),
                1.0 / gamma,
            )

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

            blended = (
                (1 - alpha) * lab_src
                + alpha * lab_dst
            )

            blended = kornia.color.lab_to_rgb(
                blended
            )

            blended = blended.movedim(1, -1)

            return torch.cat(
                (prefix, blended, suffix),
                dim=0,
            )

        elif overlap_mode == "cut":
            if overlap_side == "next_images":
                return torch.cat(
                    (
                        previous_images,
                        next_images[overlap:],
                    ),
                    dim=0,
                )

            return torch.cat(
                (
                    previous_images[:-overlap],
                    next_images,
                ),
                dim=0,
            )

        raise ValueError(
            f"Unknown overlap mode: {overlap_mode}"
        )

    # ------------------------------------------------------------------
    # AUDIO HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_audio_components(audio):
        """
        Resolve a native ComfyUI Audio mapping or VHS LazyAudioMap.

        Accessing the mapping keys causes LazyAudioMap to decode the
        underlying audio only when it is actually needed.
        """

        if audio is None:
            return None, None

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        return waveform, sample_rate

    @staticmethod
    def _resample_audio(
        waveform,
        previous_rate,
        target_rate,
    ):
        """
        Resample [1, channels, samples] audio using linear interpolation.

        Audio is normally 44.1 kHz in VHS, so this is primarily here to
        safely handle files with differing sample rates.
        """

        if previous_rate == target_rate:
            return waveform

        if waveform.shape[-1] <= 1:
            return waveform

        next_length = max(
            1,
            round(
                waveform.shape[-1]
                * target_rate
                / previous_rate
            ),
        )

        original_dtype = waveform.dtype

        # interpolate requires floating point.
        waveform = waveform.float()

        waveform = F.interpolate(
            waveform,
            size=next_length,
            mode="linear",
            align_corners=False,
        )

        return waveform.to(original_dtype)

    @staticmethod
    def _match_audio_channels(
        previous_audio,
        next_audio,
    ):
        """
        Make the next audio have the same channel count as the previous audio.

        Mono -> stereo:
            duplicate the mono channel.

        Stereo/multichannel -> mono:
            average channels.
        """

        previous_channels = previous_audio.shape[1]
        next_channels = next_audio.shape[1]

        if previous_channels == next_channels:
            return next_audio

        if previous_channels == 1:
            return next_audio.mean(
                dim=1,
                keepdim=True,
            )

        if next_channels == 1:
            return next_audio.expand(
                -1,
                previous_channels,
                -1,
            )

        # Generic fallback for unusual channel counts.
        next_audio = next_audio.mean(
            dim=1,
            keepdim=True,
        )

        return next_audio.expand(
            -1,
            previous_channels,
            -1,
        )

    @staticmethod
    def _audio_alpha(
        length,
        mode,
        device,
        dtype,
    ):
        """
        Generate the destination-side crossfade amount.

        Returned shape:
            [1, 1, samples]
        """

        if length <= 0:
            return torch.empty(
                (1, 1, 0),
                device=device,
                dtype=dtype,
            )

        t = torch.linspace(
            0,
            1,
            length + 2,
            device=device,
            dtype=dtype,
        )[1:-1]

        if mode == "linear_blend":
            alpha = t

        elif mode == "ease_in_out":
            alpha = (
                3 * t * t
                - 2 * t * t * t
            )

        elif mode == "filmic_crossfade":
            # Use a smooth nonlinear curve for audio rather than
            # applying image gamma directly to waveform amplitudes.
            gamma = 2.2

            alpha = (
                torch.pow(t, 1.0 / gamma)
            )

        elif mode == "perceptual_crossfade":
            # Equal-power crossfade.
            alpha = torch.sin(
                t * (torch.pi / 2)
            )

        else:
            alpha = t

        return alpha.view(
            1,
            1,
            -1,
        )

    # ------------------------------------------------------------------
    # AUDIO MERGING
    # ------------------------------------------------------------------

    @classmethod
    def merge_audio(
        cls,
        previous_audio,
        next_audio,
        overlap_frames,
        fps,
        overlap_side,
        overlap_mode,
    ):
        """
        Merge two audio streams using an overlap expressed in video frames.

        The audio overlap duration is:

            overlap_frames / fps

        which is then converted to samples using the previous audio
        sample rate.
        """

        previous_waveform, previous_rate = (
            cls._get_audio_components(previous_audio)
        )

        next_waveform, next_rate = (
            cls._get_audio_components(next_audio)
        )

        # If only one side contains audio, preserve the available audio.
        if previous_waveform is None:
            if next_waveform is None:
                return None

            return {
                "waveform": next_waveform,
                "sample_rate": next_rate,
            }

        if next_waveform is None:
            return {
                "waveform": previous_waveform,
                "sample_rate": previous_rate,
            }

        # Make sure both waveforms use the same sample rate.
        next_waveform = cls._resample_audio(
            next_waveform,
            next_rate,
            previous_rate,
        )

        # Make sure both waveforms use the same number of channels.
        next_waveform = cls._match_audio_channels(
            previous_waveform,
            next_waveform,
        )

        # Calculate the audio overlap from video frames.
        overlap_seconds = (
            float(overlap_frames)
            / float(fps)
        )

        overlap_samples = max(
            0,
            round(
                overlap_seconds
                * previous_rate
            ),
        )

        overlap_samples = min(
            overlap_samples,
            previous_waveform.shape[-1],
            next_waveform.shape[-1],
        )

        if overlap_samples <= 0:
            waveform = torch.cat(
                (
                    previous_waveform,
                    next_waveform,
                ),
                dim=-1,
            )

            return {
                "waveform": waveform,
                "sample_rate": previous_rate,
            }

        # --------------------------------------------------------------
        # CUT
        # --------------------------------------------------------------

        if overlap_mode == "cut":

            if overlap_side == "next_images":
                waveform = torch.cat(
                    (
                        previous_waveform,
                        next_waveform[
                            :,
                            :,
                            overlap_samples:,
                        ],
                    ),
                    dim=-1,
                )

            else:
                waveform = torch.cat(
                    (
                        previous_waveform[
                            :,
                            :,
                            :-overlap_samples,
                        ],
                        next_waveform,
                    ),
                    dim=-1,
                )

            return {
                "waveform": waveform,
                "sample_rate": previous_rate,
            }

        # --------------------------------------------------------------
        # CROSSFADE
        # --------------------------------------------------------------

        if overlap_side == "previous":
            blend_src = previous_waveform[
                :,
                :,
                -overlap_samples:,
            ]

            blend_dst = next_waveform[
                :,
                :,
                :overlap_samples,
            ]

        else:
            blend_src = next_waveform[
                :,
                :,
                :overlap_samples,
            ]

            blend_dst = previous_waveform[
                :,
                :,
                -overlap_samples:,
            ]

        alpha = cls._audio_alpha(
            overlap_samples,
            overlap_mode,
            blend_src.device,
            blend_src.dtype,
        )

        # For perceptual/equal-power crossfade, use complementary
        # sine/cosine gains rather than simply multiplying by
        # (1-alpha) and alpha.
        if overlap_mode == "perceptual_crossfade":

            t = torch.linspace(
                0,
                1,
                overlap_samples + 2,
                device=blend_src.device,
                dtype=blend_src.dtype,
            )[1:-1]

            gain_src = torch.cos(
                t * (torch.pi / 2)
            ).view(1, 1, -1)

            gain_dst = torch.sin(
                t * (torch.pi / 2)
            ).view(1, 1, -1)

            blended = (
                blend_src * gain_src
                + blend_dst * gain_dst
            )

        else:
            blended = (
                (1 - alpha) * blend_src
                + alpha * blend_dst
            )

        prefix = previous_waveform[
            :,
            :,
            :-overlap_samples,
        ]

        suffix = next_waveform[
            :,
            :,
            overlap_samples:,
        ]

        waveform = torch.cat(
            (
                prefix,
                blended,
                suffix,
            ),
            dim=-1,
        )

        return {
            "waveform": waveform,
            "sample_rate": previous_rate,
        }

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        image_batches_list,
        audio_list,
        fps,
        overlap,
        overlap_side,
        overlap_mode,
    ):
        # Because is_input_list=True, ALL inputs arrive as lists.

        if isinstance(fps, list):
            fps = fps[0]

        if isinstance(overlap, list):
            overlap = overlap[0]

        if isinstance(overlap_side, list):
            overlap_side = overlap_side[0]

        if isinstance(overlap_mode, list):
            overlap_mode = overlap_mode[0]

        fps = float(fps)
        overlap = int(overlap)

        if fps <= 0:
            raise ValueError(
                f"FPS must be greater than zero: {fps}"
            )

        if not image_batches_list:
            raise ValueError(
                "No image batches supplied"
            )

        if not audio_list:
            raise ValueError(
                "No audio supplied"
            )

        if len(image_batches_list) != len(audio_list):
            raise ValueError(
                "Image batch list and audio list must contain "
                "the same number of items: "
                f"{len(image_batches_list)} images vs "
                f"{len(audio_list)} audio"
            )

        # --------------------------------------------------------------
        # SINGLE ITEM
        # --------------------------------------------------------------

        if len(image_batches_list) == 1:
            return io.NodeOutput(
                image_batches_list[0],
                cls._audio_to_output(audio_list[0]),
            )

        # --------------------------------------------------------------
        # MERGE
        # --------------------------------------------------------------

        merged_images = image_batches_list[0]
        merged_audio = audio_list[0]

        for index in range(1, len(image_batches_list)):

            next_images = image_batches_list[index]
            next_audio = audio_list[index]

            merged_images = cls.merge_image_batches(
                merged_images,
                next_images,
                overlap,
                overlap_side,
                overlap_mode,
            )

            merged_audio = cls.merge_audio(
                merged_audio,
                next_audio,
                overlap,
                fps,
                overlap_side,
                overlap_mode,
            )

        return io.NodeOutput(
            merged_images,
            cls._audio_to_output(merged_audio),
        )

    @staticmethod
    def _audio_to_output(audio):
        """
        Normalize the final audio representation.

        A LazyAudioMap can be returned directly, but once audio has
        been merged it is already a normal waveform dictionary.
        """

        if audio is None:
            return None

        return audio
