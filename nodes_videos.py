import importlib
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io


class LoadVideosFromFolderList(io.ComfyNode):

    VIDEO_EXTENSIONS = {
        ".webm",
        ".mp4",
        ".mkv",
        ".gif",
        ".mov",
    }

    @classmethod
    def _get_vhs_loader(cls):
        module_names = (
            "ComfyUI-VideoHelperSuite.videohelpersuite.load_video_nodes",
            "comfyui-videohelpersuite.videohelpersuite.load_video_nodes",
        )

        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                loader = getattr(module, "load_video", None)

                if loader is not None:
                    return loader

            except ImportError:
                continue

        # Windows / already-loaded-module fallback.
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue

            if module_name.endswith("videohelpersuite.load_video_nodes"):
                loader = getattr(module, "load_video", None)

                if loader is not None:
                    return loader

        raise ImportError(
            "Load Videos From Folder (List) requires "
            "ComfyUI-VideoHelperSuite with load_video_nodes.load_video()."
        )

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadVideosFromFolderList",
            display_name="Load Videos From Folder List Alt ◯",
            description=(
                "Loads all supported video files from a folder and "
                "returns each video as a separate IMAGE batch, "
                "along with matching audio."
            ),
            category="image/video",
            search_aliases=[
                "load videos",
                "load video folder",
                "load videos folder",
                "video folder",
                "video list",
                "video batch list",
            ],

            inputs=[
                io.String.Input(
                    "video",
                    default="X://insert/path/",
                    tooltip="Folder containing the videos to load.",
                ),

                io.Float.Input(
                    "force_rate",
                    default=0,
                    min=0,
                    max=60,
                    step=1,
                    tooltip=(
                        "Force a specific frame rate. "
                        "0 uses the source frame rate."
                    ),
                ),

                io.Int.Input(
                    "custom_width",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Resize videos to this width. "
                        "0 preserves the source width."
                    ),
                ),

                io.Int.Input(
                    "custom_height",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Resize videos to this height. "
                        "0 preserves the source height."
                    ),
                ),

                io.Int.Input(
                    "frame_load_cap",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip=(
                        "Maximum number of frames to load from each video. "
                        "0 loads all available frames."
                    ),
                ),

                io.Int.Input(
                    "skip_first_frames",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Number of frames to skip at the beginning.",
                ),

                io.Int.Input(
                    "select_every_nth",
                    default=1,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Load every Nth frame.",
                ),

                io.Boolean.Input(
                    "add_label",
                    default=False,
                    tooltip="Add the filename above each video.",
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="image_batches_list",
                    is_output_list=True,
                ),

                io.Audio.Output(
                    display_name="audio_list",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        force_rate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        add_label,
    ):

        if not os.path.isdir(video):
            raise ValueError(
                f"Video folder does not exist or is not a directory: {video}"
            )

        videos = []

        for filename in sorted(os.listdir(video)):
            filepath = os.path.join(video, filename)

            if not os.path.isfile(filepath):
                continue

            extension = os.path.splitext(filename)[1].lower()

            if extension in cls.VIDEO_EXTENSIONS:
                videos.append(
                    (
                        filepath,
                        filename,
                    )
                )

        if not videos:
            raise ValueError(
                f"No supported video files found in folder: {video}"
            )

        vhs_load_video = cls._get_vhs_loader()

        loaded_videos = []
        loaded_audio = []

        for filepath, filename in videos:

            result = vhs_load_video(
                video=filepath,
                force_rate=force_rate,
                custom_width=custom_width,
                custom_height=custom_height,
                frame_load_cap=frame_load_cap,
                skip_first_frames=skip_first_frames,
                select_every_nth=select_every_nth,
            )

            video_tensor = result[0]
            audio = result[2]

            if add_label:
                video_tensor = cls._add_label(
                    video_tensor,
                    filename,
                )

            loaded_videos.append(video_tensor)
            loaded_audio.append(audio)

        return io.NodeOutput(
            loaded_videos,
            loaded_audio,
        )

    @staticmethod
    def _add_label(video_tensor, filename):

        if video_tensor.dim() == 4:
            _, height, width, channels = video_tensor.shape
        else:
            height, width, channels = video_tensor.shape

        label_text = os.path.splitext(filename)[0]

        font_size = max(
            16,
            width // 20,
        )

        try:
            font = ImageFont.truetype(
                "arial.ttf",
                font_size,
            )
        except OSError:
            font = ImageFont.load_default()

        dummy_img = Image.new(
            "RGB",
            (width, 10),
            (0, 0, 0),
        )

        draw = ImageDraw.Draw(dummy_img)

        text_bbox = draw.textbbox(
            (0, 0),
            label_text,
            font=font,
        )

        extra_padding = max(
            12,
            font_size // 2,
        )

        label_height = (
            text_bbox[3]
            - text_bbox[1]
            + extra_padding
        )

        label_img = Image.new(
            "RGB",
            (width, label_height),
            (0, 0, 0),
        )

        draw = ImageDraw.Draw(label_img)

        text_width = text_bbox[2] - text_bbox[0]

        draw.text(
            (
                width // 2 - text_width // 2,
                4,
            ),
            label_text,
            font=font,
            fill=(255, 255, 255),
        )

        label_np = (
            np.asarray(label_img)
            .astype(np.float32)
            / 255.0
        )

        label_tensor = torch.from_numpy(label_np)

        if channels == 1:
            label_tensor = label_tensor.mean(
                dim=2,
                keepdim=True,
            )

        elif channels == 4:
            alpha = torch.ones(
                (
                    label_height,
                    width,
                    1,
                ),
                dtype=label_tensor.dtype,
            )

            label_tensor = torch.cat(
                (
                    label_tensor,
                    alpha,
                ),
                dim=2,
            )

        if video_tensor.dim() == 4:
            label_tensor = label_tensor.unsqueeze(0).expand(
                video_tensor.shape[0],
                -1,
                -1,
                -1,
            )

            video_tensor = torch.cat(
                (
                    label_tensor,
                    video_tensor,
                ),
                dim=1,
            )

        else:
            video_tensor = torch.cat(
                (
                    label_tensor,
                    video_tensor,
                ),
                dim=0,
            )

        return video_tensor
