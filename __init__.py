from .image import SmartImageResizeAlt, MergeImageBatchList

NODE_CLASS_MAPPINGS = {"SmartImageResizeAlt": SmartImageResizeAlt,
                       "MergeImageBatchList": MergeImageBatchList,
                       }

NODE_DISPLAY_NAME_MAPPINGS = {"SmartImageResizeAlt": "🔧 Smart Image Resize Alt ◯",
                              "MergeImageBatchList": "🔧 Merge Image Batch List ◯",
                              }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
