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