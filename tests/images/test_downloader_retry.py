"""
Retry policy of ImageDownloader._download_image.

raw.githubusercontent.com answers roughly one request in three from a
mainland-China host, and the policy used to be a hardcoded 3 attempts, so an
image was lost (2/3)^3 ~= 30% of the time. These pin the behaviour the fix
depends on: transient network errors are retried up to max_attempts, and
permanent rejections are not retried at all.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from devtoolbox.images.downloader import (
    IMAGE_DOWNLOAD_MAX_ATTEMPTS,
    IMAGE_DOWNLOAD_MAX_RETRY_SECONDS,
    ImageDownloader,
)


def _downloader(**kw):
    return ImageDownloader(
        ["https://example.invalid/a.png"],
        path_prefix="",
        base_filename="t",
        storage=MagicMock(),
        **kw,
    )


@pytest.fixture(autouse=True)
def _no_waiting():
    """Collapse tenacity's 4-10s backoff so the tests run instantly."""
    with patch("devtoolbox.images.downloader.wait_exponential",
               return_value=lambda *_a, **_k: 0):
        yield


def test_transient_error_is_retried_up_to_max_attempts():
    d = _downloader(max_attempts=4)
    with patch("devtoolbox.images.downloader.requests.get",
               side_effect=requests.exceptions.ConnectTimeout) as get:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            d._download_image(0, "https://example.invalid/a.png")
    assert get.call_count == 4


def test_recovers_when_a_later_attempt_succeeds():
    """The whole point: a flaky host that answers on the third try."""
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    ok = MagicMock(headers={"content-type": "image/png"}, content=png)
    d = _downloader(max_attempts=6)
    with patch("devtoolbox.images.downloader.requests.get",
               side_effect=[requests.exceptions.ConnectionError,
                            requests.exceptions.ConnectTimeout,
                            ok]) as get, \
            patch("devtoolbox.images.downloader.Image.open") as img, \
            patch("devtoolbox.images.downloader.imagehash.dhash",
                  return_value="deadbeef"):
        img.return_value.size = (800, 600)
        d._download_image(0, "https://example.invalid/a.png")
    assert get.call_count == 3


def test_non_image_content_type_is_not_retried():
    """
    A 404 page or an HTML error body is permanent. Retrying it would spend
    the whole backoff budget on something that can never succeed.
    """
    resp = MagicMock(headers={"content-type": "text/html"}, content=b"nope")
    d = _downloader(max_attempts=6)
    with patch("devtoolbox.images.downloader.requests.get",
               return_value=resp) as get:
        out = d._download_image(0, "https://example.invalid/a.png")
    assert get.call_count == 1
    assert out["content"] is None


def test_default_attempts_come_from_the_module_constant():
    assert _downloader().max_attempts == IMAGE_DOWNLOAD_MAX_ATTEMPTS
    assert IMAGE_DOWNLOAD_MAX_ATTEMPTS >= 6


def test_wall_clock_ceiling_stops_a_dead_host_early():
    """
    The attempt count alone does not bound elapsed time. A host that always
    times out must not be able to spend max_attempts x timeout + backoff --
    with COLLECTOR_WORKERS = 2 that is minutes per project, against callers
    whose whole collection runs under one lock.
    """
    d = _downloader(max_attempts=50, max_retry_seconds=0)
    with patch("devtoolbox.images.downloader.requests.get",
               side_effect=requests.exceptions.ConnectTimeout) as get:
        with pytest.raises(requests.exceptions.ConnectTimeout):
            d._download_image(0, "https://example.invalid/a.png")
    # stop_after_delay(0) trips immediately, so the 50 attempts never happen.
    assert get.call_count == 1


def test_ceiling_defaults_leave_room_for_several_attempts():
    """The ceiling must not be so tight that it cancels the extra attempts."""
    assert IMAGE_DOWNLOAD_MAX_RETRY_SECONDS >= 30
    assert _downloader().max_retry_seconds == IMAGE_DOWNLOAD_MAX_RETRY_SECONDS
