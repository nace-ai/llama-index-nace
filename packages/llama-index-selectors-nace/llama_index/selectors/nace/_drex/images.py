"""Convert images to the entries Drex reads in a request's `images` array."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from typing import Any

import nace_sdk

from .errors import NaceImageError

_REMOTE = ("http://", "https://")

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def _remote_url_error(url: str) -> NaceImageError:
    return NaceImageError(
        f"Drex doesn't fetch image URLs ({url[:60]}…). Download the image and pass its "
        "bytes, or upload it with `nace_sdk` and pass the uploaded file."
    )


def as_data_url(image: str) -> str:
    """A data URL for base64 image bytes, typed from their first bytes."""
    if image.startswith("data:"):
        return image
    head = base64.b64decode(image[:24] + "=" * (-len(image[:24]) % 4), validate=False)
    mime = next((m for sig, m in _SIGNATURES if head.startswith(sig)), None)
    if mime is None and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        mime = "image/webp"
    return f"data:{mime or 'application/octet-stream'};base64,{image}"


def _is_base64(value: str) -> bool:
    if len(value) % 4:
        return False
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def to_drex_image(
    item: str
    | bytes
    | os.PathLike[str]
    | nace_sdk.UploadedFile
    | nace_sdk.WorkspaceFileSource
    | Mapping[str, str],
) -> str | dict[str, Any]:
    """Return one `images` entry for Drex.

    Accepts base64, a data URL, bytes, a path, a `workspace_file` reference, or a
    `nace_sdk` uploaded file. Remote URLs are refused: Drex never fetches them.

    Raises:
        NaceImageError: The item isn't an image Drex can read.

    """
    if isinstance(item, str):
        if item.startswith("data:"):
            return item
        if item.startswith(_REMOTE):
            raise _remote_url_error(item)
        # A path first: a relative path such as "img/blue" can also be valid base64.
        if os.path.exists(item):
            return _from_sdk(item)
        if _is_base64(item):
            return item
        message = f"{item[:60]!r} is not base64, a data URL, or an existing file path."
        raise NaceImageError(message)
    if isinstance(item, bytes | bytearray | os.PathLike):
        return _from_sdk(item)
    if isinstance(item, nace_sdk.UploadedFile | nace_sdk.WorkspaceFileSource):
        return _from_sdk(item)
    if isinstance(item, Mapping):
        if item.get("type") == "workspace_file":
            return dict(item)
        message = f"Unsupported image block: {dict(item)!r:.80}"
        raise NaceImageError(message)
    message = f"Unsupported image type: {type(item).__name__}."
    raise NaceImageError(message)


def _from_sdk(item: Any) -> str | dict[str, Any]:
    try:
        return nace_sdk.image_input(item)
    except (nace_sdk.NaceError, OSError) as error:
        raise NaceImageError(f"Can't read the image: {type(error).__name__}.") from error


__all__ = ["as_data_url", "to_drex_image"]
