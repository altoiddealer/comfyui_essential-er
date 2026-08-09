from comfy_api.latest import ComfyExtension

from .nodes_image import SmartImageResizeAlt
from .nodes_videos import LoadVideosFromFolderList
from .nodes_batching import MergeImageBatchList, MergeImageBatchAndAudioList


class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [
            SmartImageResizeAlt,
            LoadVideosFromFolderList,
            MergeImageBatchList,
            MergeImageBatchAndAudioList,
        ]


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()