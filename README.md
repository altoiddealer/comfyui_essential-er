# 🔧 ComfyUI Essential-er

---

Enhanced versions of existing nodes.

---

## Included Nodes

### Resize Image/Mask Alt

Customized version of native ComfyUI node **Resize Image/Mask**

| Feature                     | **ResizeImageMaskAlt**                                                                                   | **ComfyUI `ResizeImageMask`**                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Image & Mask Input**      | Can resize an **Image, Mask, or both simultaneously**                                                    | Resizes an **Image or Mask**, but not both simultaneously                                    |
| **Output Dimensions**       | Can constrain dimensions to **multiples of** a specified value                                           | No **multiple-of** dimension constraint                                                      |
| **Resize Condition**        | Optional **condition** can prevent unwanted resizing during batch processing                             | **Always resizes**; no conditional filtering                                                 |
| **Aspect Ratio / Cropping** | **Configurable crop method** for all aspect-ratio mismatches                                             | Crop behavior **sometimes** configurable / hardcoded for some resize types                   |
| **"Smart Resize" Methods**  | Scale to target megapixels or resolution while conforming to the source or a selected aspect ratio       | Not available                                                                                |
| **Pad**                     | Built-in **Pad** method with selectable **black, grey, or white** padding                                | No Pad method; ComfyUI provides a separate **Resize And Pad Image** node with fewer controls |
| **Combined Controls**       | Resize, crop, pad, conditions, and dimension constraints are available within one node                   | Functionality is split between multiple nodes and has fewer configuration options            |


<img width="1151" height="604" alt="Screenshot 2026-08-14 163454" src="https://github.com/user-attachments/assets/9fb6438c-ad0a-4f93-a8e2-a72d0c1bf11f" />


---

### Load Videos From Folder List

Based on **Load Videos From Folder** from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).

KJ's version outputs the merged images (all loaded videos are joined), which has very limited practicality.

My version outputs both the video images ***and*** audio in lists which can be more easily processed.

The following nodes are designed specifically to handle the outputs from **Load Videos From Folder List**.


### Merge Image Batch List

Based on **Image Batch Extend With Overlap** from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), which is used to join separate videos (image batches) using configurable joining strategies, and outputting an extended video.

**Merge Image Batch List** applies the same joining logic, but expects a single input which is a list of image batches. This node will iterate over the list and join all the videos together using the same joining strategy.

This can greatly simplify use cases for joining many similar clips together, where any numeber of clips can be loaded from a directory, Rebatched, and piped into this node, instead of manually loaded and daisy-chained into **Image Batch Extend With Overlap** nodes.


### Merge Image Batches and Audio Lists

Same as above, but also handles audio!  The audio must be sourced from videos that share the same FPS, and that FPS must be specified in the node input widget.

<img width="1577" height="888" alt="Screenshot 2026-08-08 232456" src="https://github.com/user-attachments/assets/5c59ec5b-61a2-4597-8adb-f12a9b80aeb2" />

<details>
  <summary>DEPRECATED NODES</summary>

  **Smart Image Resize Alt [Deprecated]**

  Customized version of **Image Resize+** from [ComfyUI-essentials](https://github.com/cubiq/ComfyUI_essentials)

  The only difference between **Smart Image Resize Alt** and **Image Resize+** is the logic for the **`keep_proportions`** option.

  In **Smart Image Resize Alt** the result has the closest matching megapixels count to the average **width** and **height** values - instead of making one dimension match an input and scaling the other down (which results in undesirable resolutions)

  <img width="1037" height="574" alt="Screenshot 2025-07-27 134150" src="https://github.com/user-attachments/assets/5d27b7b5-9be6-40ca-ba9a-e1874afc65a3" />

</details>
