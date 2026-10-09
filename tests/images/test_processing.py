"""Byte-oriented image processing does not fetch URLs or write files."""

from io import BytesIO

import pytest
from PIL import Image

from devtoolbox.images.processing import (
    ImageProcessingError,
    ImageRejected,
    detect_image_mime,
    inspect_image_bytes,
    normalize_image_bytes,
)


def _image_bytes(size=(800, 600), *, format="JPEG", color="blue"):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=format)
    return output.getvalue()


def test_detect_image_type_from_bytes():
    assert detect_image_mime(_image_bytes(format="WEBP")) == "image/webp"
    with pytest.raises(ImageRejected, match="invalid"):
        detect_image_mime(b"<svg/>")


def test_inspect_filters_before_returning_hash():
    info = inspect_image_bytes(_image_bytes())
    assert (info.mime_type, info.width, info.height) == (
        "image/jpeg", 800, 600
    )
    assert len(info.perceptual_hash) == 16
    with pytest.raises(ImageRejected, match="size or aspect ratio"):
        inspect_image_bytes(_image_bytes((120, 120)))
    with pytest.raises(ImageRejected, match="size or aspect ratio"):
        inspect_image_bytes(_image_bytes((1200, 600)))
    with pytest.raises(ImageRejected, match="safety limit"):
        inspect_image_bytes(_image_bytes(), max_pixels=100)
    with pytest.raises(ImageRejected, match="unsupported"):
        inspect_image_bytes(
            _image_bytes(), allowed_mime_types={"image/webp"}
        )


def test_normalize_resizes_orients_and_flattens_transparency():
    source = Image.new("RGBA", (1600, 800), (255, 0, 0, 0))
    source.putpixel((500, 500), (0, 0, 255, 255))
    original = BytesIO()
    source.save(original, format="PNG")
    normalized = normalize_image_bytes(
        original.getvalue(), max_size=(1280, 1280)
    )
    with Image.open(BytesIO(normalized)) as result:
        assert result.format == "WEBP"
        assert result.size == (1280, 640)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)

    oriented = Image.new("RGB", (800, 600), "blue")
    exif = Image.Exif()
    exif[274] = 6
    original = BytesIO()
    oriented.save(original, format="JPEG", exif=exif)
    with Image.open(BytesIO(normalize_image_bytes(original.getvalue()))) as result:
        assert result.size == (600, 800)


def test_invalid_bytes_and_output_limit_have_distinct_errors():
    with pytest.raises(ImageRejected, match="invalid"):
        normalize_image_bytes(b"not an image")
    with pytest.raises(ImageProcessingError, match="size limit"):
        normalize_image_bytes(_image_bytes(), max_output_bytes=1)
