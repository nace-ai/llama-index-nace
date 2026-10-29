"""Turn retrieved source nodes into Drex contexts and images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import ImageNode, NodeWithScore

from ._drex.errors import NaceImageError, NaceImageLimitError
from ._drex.images import as_data_url, to_drex_image

MAX_IMAGES = 8


def _node_image(node: ImageNode) -> str | dict[str, Any]:
    if node.image:
        if node.image.startswith("data:"):
            return node.image
        if node.image_mimetype:
            return f"data:{node.image_mimetype};base64,{node.image}"
        return as_data_url(node.image)
    if node.image_path:
        return to_drex_image(node.image_path)
    if node.image_url:
        return to_drex_image(node.image_url)
    message = "An ImageNode in the sources has no image, image_path or image_url."
    raise NaceImageError(message)


def _keep_top(nodes: list[NodeWithScore], max_images: int | None) -> list[NodeWithScore]:
    images = [n for n in nodes if isinstance(n.node, ImageNode)]
    if max_images is None:
        if len(images) <= MAX_IMAGES:
            return nodes
        message = (
            f"{len(images)} image nodes in the sources, but Drex reads at most {MAX_IMAGES} "
            "images per request. Set `max_images` to keep the highest-scored ones."
        )
        raise NaceImageLimitError(message)
    if len(images) <= max_images:
        return nodes
    ranked = sorted(images, key=lambda n: -(n.score if n.score is not None else float("-inf")))
    dropped = {id(n) for n in ranked[max_images:]}
    return [n for n in nodes if id(n) not in dropped]


def split_sources(
    nodes: Sequence[NodeWithScore], *, max_images: int | None
) -> tuple[list[str], list[str | dict[str, Any]]]:
    """Split source nodes into `(contexts, images)`.

    Text nodes become contexts. Image nodes become images, and their text, if any, is added
    to the contexts as `"[Image N] <text>"`. With `max_images` set, only the `max_images`
    highest-scored image nodes are kept, whatever their count. Without it, more than 8 images
    are refused rather than silently trimmed.

    Raises:
        NaceImageError: An image node has no readable source, or only an http(s) URL.
        NaceImageLimitError: More than 8 images and no `max_images`.

    """
    contexts: list[str] = []
    images: list[str | dict[str, Any]] = []
    for item in _keep_top(list(nodes), max_images):
        node = item.node
        if isinstance(node, ImageNode):
            images.append(_node_image(node))
            if node.text:
                contexts.append(f"[Image {len(images)}] {node.text}")
        else:
            contexts.append(node.get_content())
    return contexts, images
