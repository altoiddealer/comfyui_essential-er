# 🔧 ComfyUI Essential-er

---

Slightly enhanced versions of [ComfyUI-essentials](https://github.com/cubiq/ComfyUI_essentials) nodes.

> Essential nodes that are weirdly missing from ComfyUI core. With few exceptions they are new features and not commodities. I hope this will be just a temporary repository until the nodes get included into ComfyUI.

---

## Included Nodes

### Smart Image Resize

The only difference between **Smart Image Resize** and **Image Resize+** is the logic for the **`keep_proportions`** option.

In **Smart Image Resize** the result has the closest matching megapixels count to the average **width** and **height** values - instead of making one dimension match an input and scaling the other down (which results in undesirable resolutions)

<img width="1037" height="574" alt="Screenshot 2025-07-27 134150" src="https://github.com/user-attachments/assets/5d27b7b5-9be6-40ca-ba9a-e1874afc65a3" />
