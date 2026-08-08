from comfy_api.latest import ComfyExtension

from .image import SmartImageResizeAlt


class SmartImageResizeAltExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [SmartImageResizeAlt]


async def comfy_entrypoint() -> SmartImageResizeAltExtension:
    return SmartImageResizeAltExtension()
