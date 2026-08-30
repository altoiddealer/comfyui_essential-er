from comfy_api.latest import ComfyExtension

from .nodes_image import ResizeImageMaskAlt, SmartImageResizeAlt
from .nodes_videos import LoadVideosFromFolderList
from .nodes_batching import MergeImageBatchList, MergeImageBatchAndAudioList
from .nodes_utility import PassOrNone

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [
            ResizeImageMaskAlt,
            SmartImageResizeAlt,
            LoadVideosFromFolderList,
            MergeImageBatchList,
            MergeImageBatchAndAudioList,
            PassOrNone,
        ]


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()