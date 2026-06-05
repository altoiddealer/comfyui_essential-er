# 🔧 ComfyUI Essential-er

---

Enhanced versions of existing nodes.

---

## Included Nodes

### Smart Image Resize Alt

Customized version of **Image Resize+** from [ComfyUI-essentials](https://github.com/cubiq/ComfyUI_essentials)

The only difference between **Smart Image Resize Alt** and **Image Resize+** is the logic for the **`keep_proportions`** option.

In **Smart Image Resize Alt** the result has the closest matching megapixels count to the average **width** and **height** values - instead of making one dimension match an input and scaling the other down (which results in undesirable resolutions)

<img width="1037" height="574" alt="Screenshot 2025-07-27 134150" src="https://github.com/user-attachments/assets/5d27b7b5-9be6-40ca-ba9a-e1874afc65a3" />

### Merge Image Batch List

Based on **Image Batch Extend With Overlap** from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), which is used to join separate videos (image batches) using configurable joining strategies, and outputting an extended video.

**Merge Image Batch List** applies the same joining logic, but expects a single input which is a list of image batches. This node will iterate over the list and join all the videos together using the same joining strategy.

This can greatly simplify use cases for joining many similar clips together, where any numeber of clips can be loaded from a directory, Rebatched, and piped into this node, instead of manually loaded and daisy-chained into **Image Batch Extend With Overlap** nodes.

<img width="1019" height="328" alt="Screenshot 2026-06-05 121610" src="https://github.com/user-attachments/assets/bfcbc9b3-f430-439d-a943-8bd626608c8b" />

