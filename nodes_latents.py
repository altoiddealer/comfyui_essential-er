from comfy_api.latest import io
import os
from safetensors.torch import load_file, save_file
import folder_paths
import logging
_LOG = logging.getLogger("comfui-essential-er")

"""
custom nodes for saving and loading MiniMax H3 audio/video
NestedTensor latents as .safetensors files.

Save:
    H3 NestedTensor LATENT
        -> video + audio tensors
        -> .safetensors

Load:
    .safetensors
        -> video + audio tensors
        -> ComfyUI NestedTensor
        -> LATENT

Load paths may be:

    - absolute file path
    - absolute directory
    - path relative to ComfyUI output directory
    - directory relative to ComfyUI output directory

For directories:

    clip_index == 0
        Load newest .safetensors.

    clip_index > 0
        Load deterministic clip slot:

            *_00001.safetensors
            *_00002.safetensors
            ...

        Auto-numbered files ending with an additional underscore:

            *_00001_.safetensors

        are deliberately ignored for indexed loads.
"""

def _streams_from_latent(latent):
    """
    Extract the constituent video/audio tensors from an H3 AV latent.

    H3's samples are a ComfyUI NestedTensor containing:
        [video, audio]

    The two tensors have different dimensionality, so they cannot simply
    be stacked into an ordinary tensor.
    """

    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("ComfyUI-essential-er: expected a LATENT containing a 'samples' field.")

    samples = latent["samples"]

    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())

    elif isinstance(samples, (list, tuple)):
        parts = list(samples)

    else:
        raise ValueError(
            "ComfyUI-essential-er: expected a NestedTensor containing the H3 video/audio streams, got %r."
            % type(samples)
        )

    if len(parts) != 2:
        raise ValueError(
            "ComfyUI-essential-er: expected exactly 2 streams (video and audio), got %d."
            % len(parts)
        )

    return parts[0], parts[1]

def _rebuild_nested(video, audio):
    """
    Reconstruct ComfyUI's H3 NestedTensor.

    This is intentionally ComfyUI's own NestedTensor implementation
    rather than torch.nested.nested_tensor(), because the native H3
    VAE decoder expects ComfyUI's NestedTensor object.
    """

    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor(
        [
            video,
            audio,
        ]
    )

def _resolve_latent_path(path, clip_index=0):
    """
    Turn the loader's path input into a concrete file.

    Accepts:

        - absolute path
        - path relative to ComfyUI's output folder
        - directory in either form

    For a directory:

        clip_index == 0
            The NEWEST .safetensors inside is used.

            This is simple, but NOT retry-safe: re-rolling a clip can
            cause the rejected attempt to become the newest save.

        clip_index > 0
            Exactly that clip's slot is loaded:

                clip 1 -> *_00001.safetensors
                clip 2 -> *_00002.safetensors

            Auto-numbered files carry a trailing underscore:

                *_00001_.safetensors

            and are deliberately NOT matched for indexed loads.
    """

    p = (path or "").strip().strip('"').strip("'")

    if not p:
        p = "h3_context"

    output_directory = folder_paths.get_output_directory()

    candidates = [
        p,
        os.path.join(output_directory, p),
    ]

    for c in candidates:

        # Explicit file
        if os.path.isfile(c):
            return c

        # Directory
        if os.path.isdir(c):
            idx = int(clip_index)
            entries = os.listdir(c)

            # Deterministic indexed clip
            if idx > 0:
                endings = (
                    "_%05d.safetensors" % idx,
                    "_clip%03d.safetensors" % idx,
                )
                files = [
                    os.path.join(c, filename)
                    for filename in entries
                    if filename.endswith(endings)
                ]

                if not files:
                    # Check whether the user has the auto-numbered
                    # variant and provide a useful diagnostic.
                    near = [
                        filename
                        for filename in entries
                        if filename.endswith(
                            "_%05d_.safetensors" % idx
                        )
                    ]

                    hint = ""

                    if near:
                        hint = (
                            " Found %s, which is an auto-numbered save "
                            "(trailing underscore = numbered by RUN, so "
                            "it may be a rejected attempt). If it really "
                            "is clip %d, rename it to drop the trailing "
                            "underscore: %s"
                            % (
                                near[0],
                                idx,
                                near[0].replace(
                                    "_%05d_" % idx,
                                    "_%05d" % idx,
                                ),
                            )
                        )

                    raise FileNotFoundError(
                        "ComfyUI-essential-er: no saved latent for clip %d "
                        "(no *_%05d.safetensors in %s).%s"
                        % (
                            idx,
                            idx,
                            c,
                            hint,
                        )
                    )

                return max(
                    files,
                    key=os.path.getmtime,
                )

            # ---------------------------------------------------------
            # clip_index == 0 -> newest file
            # ---------------------------------------------------------

            files = [
                os.path.join(c, filename)
                for filename in entries
                if filename.endswith(".safetensors")
            ]

            if not files:
                raise FileNotFoundError(
                    "ComfyUI-essential-er: no .safetensors latents found in %s. "
                    "Run a clip with Save H3 AV Latent first."
                    % c
                )

            return max(
                files,
                key=os.path.getmtime,
            )

    raise FileNotFoundError(
        "ComfyUI-essential-er: %r is neither a file nor a folder "
        "(also tried relative to the ComfyUI output directory)."
        % p
    )


class SaveH3AVLatentAlt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveH3AVLatentAlt",
            display_name="Save H3 AV Latent Alt ◯",
            category="latent/H3",
            description="Save a MiniMax H3 audio/video NestedTensor latent as a .safetensors file.",

            inputs=[
                io.Latent.Input(
                    "samples",
                    display_name="latent",
                    tooltip= "MiniMax H3 sampler output containing the video/audio NestedTensor.",
                ),

                io.String.Input(
                    "filename_prefix",
                    default="h3_context/clip",
                    tooltip="Output directory and filename prefix. Relative paths are resolved inside ComfyUI's output directory.",
                ),

                io.Int.Input(
                    "clip_index",
                    default=0,
                    min=0,
                    max=999999,
                    tooltip=(
                        "0 = automatically number the save by RUN. "
                        "A positive value creates a deterministic clip slot such as *_00001.safetensors."
                    ),
                ),
            ],

            outputs=[
                io.Latent.Output(
                    "latent",
                    display_name="latent",
                    tooltip="The original H3 AV latent, passed through unchanged.",
                ),

                io.String.Output(
                    "path",
                    display_name="saved path",
                    tooltip="Absolute path of the saved .safetensors file.",
                ),
            ],

            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        samples,
        filename_prefix,
        clip_index,
    ):

        video, audio = _streams_from_latent(samples)

        # Safetensors requires regular contiguous CPU tensors.
        video = (
            video
            .detach()
            .cpu()
            .contiguous()
        )

        audio = (
            audio
            .detach()
            .cpu()
            .contiguous()
        )

        (
            folder,
            filename,
            counter,
            _subfolder,
            _filename,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
        )

        os.makedirs(
            folder,
            exist_ok=True,
        )

        # Deterministic clip slot
        if int(clip_index) > 0:
            output_path = os.path.join(
                folder,
                "%s_%05d.safetensors"
                % (
                    filename,
                    int(clip_index),
                ),
            )

        # Auto-numbered attempt
        else:
            output_path = os.path.join(
                folder,
                "%s_%05d_.safetensors"
                % (
                    filename,
                    counter,
                ),
            )

        metadata = {
            "format": "h3_av_latent_v1",
            "nested": "true",
            "parts": "2",
            "video_shape": str(tuple(video.shape)),
            "audio_shape": str(tuple(audio.shape)),
        }

        save_file(
            {
                "video": video,
                "audio": audio,
            },
            output_path,
            metadata=metadata,
        )

        _LOG.info(
            "ComfyUI-essential-er: saved AV latent to %s "
            "(video=%s, audio=%s)",
            output_path,
            tuple(video.shape),
            tuple(audio.shape),
        )

        return io.NodeOutput(
            samples,
            output_path,
        )


class LoadH3AVLatentAlt(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadH3AVLatentAlt",
            display_name="Load H3 AV Latent Alt ◯",
            category="latent/H3",
            description="Load a MiniMax H3 AV NestedTensor latent from a .safetensors file or directory.",
            inputs=[
                io.String.Input(
                    "latent_path",
                    default="h3_context",
                    tooltip=(
                        "A .safetensors file or directory. "
                        "Absolute paths are accepted. Relative paths "
                        "are resolved against ComfyUI's output directory."
                    ),
                ),
                io.Int.Input(
                    "clip_index",
                    default=0,
                    min=0,
                    max=999999,
                    tooltip=(
                        "When latent_path is a directory: "
                        "0 loads the newest .safetensors; "
                        "a positive value loads that exact clip slot, "
                        "such as *_00001.safetensors."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    display_name="latent",
                    tooltip=(
                        "Reconstructed MiniMax H3 NestedTensor LATENT. "
                        "Can be connected to VAE Decode or H3 Motion Context."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent_path,
        clip_index,
    ):

        path = _resolve_latent_path(
            latent_path,
            clip_index,
        )

        tensors = load_file(path)

        if "video" not in tensors:
            raise ValueError(
                "ComfyUI-essential-er: %s does not contain a 'video' tensor."
                % path
            )

        if "audio" not in tensors:
            raise ValueError(
                "ComfyUI-essential-er: %s does not contain an 'audio' tensor."
                % path
            )

        video = tensors["video"]
        audio = tensors["audio"]

        nested_samples = _rebuild_nested(
            video,
            audio,
        )

        latent = {
            "samples": nested_samples,
        }

        _LOG.info(
            "ComfyUI-essential-er: loaded AV latent from %s "
            "(video=%s, audio=%s)",
            path,
            tuple(video.shape),
            tuple(audio.shape),
        )

        return io.NodeOutput(
            latent,
        )
