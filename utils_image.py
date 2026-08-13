from math import sqrt, gcd

# ------------------------------------------------------------------
# General dimension helpers
# ------------------------------------------------------------------

def round_to_multiple(value, multiple):
    """
    Round a dimension to the nearest multiple.
    """
    value = int(round(value))

    if multiple <= 1:
        return max(1, value)

    return max(
        multiple,
        int(round(value / multiple)) * multiple,
    )

def round_dimensions_to_multiple(
    width,
    height,
    multiple,
):
    """
    Round both dimensions independently to the nearest multiple.
    """
    if multiple <= 1:
        return (
            max(1, int(round(width))),
            max(1, int(round(height))),
        )

    return (
        round_to_multiple(width, multiple),
        round_to_multiple(height, multiple),
    )

def floor_to_multiple(value, multiple):
    """
    Floor a dimension to a multiple.
    """
    value = int(value)

    if multiple <= 1:
        return max(1, value)

    return max(
        multiple,
        (value // multiple) * multiple,
    )

# ------------------------------------------------------------------
# Image / Mask helpers
# ------------------------------------------------------------------

def is_image_tensor(input_tensor):
    """
    IMAGE:
        [B, H, W, C]

    MASK:
        [B, H, W]
    """
    return len(input_tensor.shape) == 4

def image_mask_to_nchw(input_tensor):
    """
    Convert a ComfyUI IMAGE or MASK tensor to [B, C, H, W].
    """
    if input_tensor.ndim == 4:
        return input_tensor.movedim(-1, 1)

    if input_tensor.ndim == 3:
        return input_tensor.unsqueeze(1)

    raise ValueError(
        f"Expected IMAGE [B,H,W,C] or MASK [B,H,W], "
        f"got shape {tuple(input_tensor.shape)}."
    )

def nchw_to_image_mask(
    input_tensor,
    is_type_image,
):
    """
    Convert [B, C, H, W] back to the original ComfyUI IMAGE/MASK layout.
    """
    if is_type_image:
        return input_tensor.movedim(1, -1)

    return input_tensor.squeeze(1)

def image_mask_dimensions(input_tensor):
    """
    Return width and height for a ComfyUI IMAGE or MASK tensor.
    """
    if input_tensor.ndim == 4:
        _, height, width, _ = input_tensor.shape

    elif input_tensor.ndim == 3:
        _, height, width = input_tensor.shape

    else:
        raise ValueError(
            f"Expected IMAGE [B,H,W,C] or MASK [B,H,W], "
            f"got shape {tuple(input_tensor.shape)}."
        )

    return width, height

# ------------------------------------------------------------------
# Aspect Ratio helpers
# ------------------------------------------------------------------

def round_to_precision(val, prec):
    return round(val / prec) * prec

def res_to_model_fit(avg, w, h, prec):
    mp = w * h
    mp_target = avg * avg
    scale = sqrt(mp_target / mp)
    w = int(round_to_precision(w * scale, prec))
    h = int(round_to_precision(h * scale, prec))
    return w, h

def dims_from_ar(avg, n, d, prec=64):
    prec = max(1, int(prec))
    doubleavg = avg * 2
    ar_sum = n+d
    # calculate width and height by factoring average with aspect ratio
    w = round((n / ar_sum) * doubleavg)
    h = round((d / ar_sum) * doubleavg)
    # Round to correct megapixel precision
    w, h = res_to_model_fit(avg, w, h, prec)
    return w, h

def avg_from_dims(w, h):
    avg = (w + h) // 2
    if (w + h) % 2 != 0:
        avg += 1
    return avg

def ar_parts_from_dims(w, h):
    divisor = gcd(w, h)
    simp_w = w // divisor
    simp_h = h // divisor
    return simp_w, simp_h

def ar_parts_from_str(ar_str: str):
    """
    Parse an aspect-ratio option such as "16:9 (Widescreen) into:(16, 9)
    """
    if not ar_str:
        raise ValueError("Aspect ratio cannot be empty.")

    ratio_text = (str(ar_str).strip().split()[0])

    if ":" not in ratio_text:
        raise ValueError(f"Invalid aspect ratio: {ar_str}")

    numerator, denominator = (ratio_text.split(":", 1))

    try:
        numerator = int(numerator)
        denominator = int(denominator)

    except ValueError as error:

        raise ValueError(f"Invalid aspect ratio: {ar_str}") from error

    if (numerator <= 0
        or denominator <= 0):
        raise ValueError(f"Aspect ratio values must be positive: {ar_str}")

    divisor = gcd(numerator, denominator)

    return (numerator // divisor,
            denominator // divisor)

# ------------------------------------------------------------------
# Misc helpers
# ------------------------------------------------------------------

@staticmethod
def pad_color_value(color):
    """
    Convert a pad color name to the normalized tensor value used by
    ComfyUI IMAGE and MASK tensors.
    """

    values = {
        "black": 0.0,
        "gray": 0.5,
        "white": 1.0,
    }

    try:
        return values[color]
    except KeyError:
        raise ValueError(
            f"Unsupported pad color: {color}"
        )