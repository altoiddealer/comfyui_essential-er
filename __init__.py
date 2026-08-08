from comfy_api.latest import ComfyExtension

from .image import SmartImageResizeAlt
from .image_batches import MergeImageBatchList


class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [
            SmartImageResizeAlt,
            MergeImageBatchList,
        ]


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()