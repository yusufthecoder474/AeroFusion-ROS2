"""Pure-Python synthetic RGB8 frames; no camera, model, or ROS dependency."""

def generate_rgb_frame(width: int, height: int, frame_index: int) -> bytes:
    if width < 64 or height < 64:
        raise ValueError('Image dimensions must be at least 64x64')
    if frame_index < 0:
        raise ValueError('frame_index must be non-negative')

    background = bytes((26, 36, 56))
    moving_box = bytes((245, 55, 55))
    image = bytearray(background * (width * height))
    size = 32
    x0 = 8 + (frame_index * 8) % (width - size - 16)
    y0 = height // 2 - size // 2
    strip = moving_box * size
    for y in range(y0, y0 + size):
        offset = (y * width + x0) * 3
        image[offset : offset + len(strip)] = strip
    return bytes(image)
