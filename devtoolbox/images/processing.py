"""Validate and normalize image bytes already obtained by a caller."""

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Collection, Mapping, Optional, Tuple

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from devtoolbox.images.downloader import (
    ASPECT_RATIO_RANGE,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)


class ImageRejected(ValueError):
    """The supplied bytes are not a usable image for the requested policy."""


class ImageProcessingError(RuntimeError):
    """A valid image could not be converted or bounded as requested."""


@dataclass(frozen=True)
class ImageInfo:
    mime_type: str
    width: int
    height: int
    perceptual_hash: str


def detect_image_mime(data: bytes) -> str:
    """Identify a raster image from its bytes, independent of HTTP headers."""
    if not isinstance(data, bytes) or not data:
        raise ImageRejected("Image bytes are empty")
    try:
        with Image.open(BytesIO(data)) as image:
            mime_type = Image.MIME.get(image.format, "")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected("Image bytes are invalid") from exc
    if not mime_type or not mime_type.startswith("image/"):
        raise ImageRejected("Image format is unsupported")
    return mime_type


def inspect_image_bytes(
    data: bytes,
    *,
    min_width: int = MIN_IMAGE_WIDTH,
    min_height: int = MIN_IMAGE_HEIGHT,
    aspect_ratio_range: Tuple[float, float] = ASPECT_RATIO_RANGE,
    max_pixels: int = 40_000_000,
    allowed_mime_types: Optional[Collection[str]] = None,
) -> ImageInfo:
    """Check size and format before hashing an image for caller-side dedupe."""
    if not isinstance(data, bytes) or not data:
        raise ImageRejected("Image bytes are empty")
    try:
        with Image.open(BytesIO(data)) as image:
            mime_type = Image.MIME.get(image.format, "")
            width, height = image.size
            if not mime_type or (
                allowed_mime_types is not None
                and mime_type not in allowed_mime_types
            ):
                raise ImageRejected("Image format is unsupported")
            if width * height > max_pixels:
                raise ImageRejected("Image dimensions exceed the safety limit")
            if width <= 0 or height <= 0:
                raise ImageRejected("Image dimensions are invalid")
            ratio = width / height
            if (
                width < min_width
                or height < min_height
                or not aspect_ratio_range[0] < ratio < aspect_ratio_range[1]
            ):
                raise ImageRejected(
                    "Image does not meet size or aspect ratio requirements"
                )
            image.load()
            perceptual_hash = str(imagehash.dhash(image))
    except ImageRejected:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected("Image bytes are invalid") from exc
    return ImageInfo(mime_type, width, height, perceptual_hash)


def normalize_image_bytes(
    data: bytes,
    *,
    max_size: Tuple[int, int] = (1280, 1280),
    output_format: str = "WEBP",
    quality: int = 78,
    save_options: Optional[Mapping[str, Any]] = None,
    max_output_bytes: Optional[int] = None,
    max_pixels: int = 40_000_000,
) -> bytes:
    """Apply EXIF orientation, resize, flatten alpha, and encode in memory."""
    if not isinstance(data, bytes) or not data:
        raise ImageRejected("Image bytes are empty")
    if min(max_size) <= 0 or not 1 <= quality <= 100:
        raise ValueError("Invalid image conversion settings")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise ImageRejected("Image dimensions exceed the safety limit")
            source.load()
            try:
                converted = ImageOps.exif_transpose(source)
                converted.thumbnail(max_size, Image.Resampling.LANCZOS)
                if converted.mode in ("RGBA", "LA") or "transparency" in converted.info:
                    rgba = converted.convert("RGBA")
                    canvas = Image.new("RGB", converted.size, "white")
                    canvas.paste(rgba, mask=rgba.getchannel("A"))
                    converted = canvas
                else:
                    converted = converted.convert("RGB")
                output = BytesIO()
                converted.save(
                    output, format=output_format, quality=quality,
                    **dict(save_options or {})
                )
                result = output.getvalue()
            except (OSError, ValueError) as exc:
                raise ImageProcessingError("Image conversion failed") from exc
    except ImageRejected:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected("Image bytes are invalid") from exc
    if not result or (max_output_bytes is not None and len(result) > max_output_bytes):
        raise ImageProcessingError("Normalized image exceeds the size limit")
    return result
