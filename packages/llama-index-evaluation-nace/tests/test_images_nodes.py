from __future__ import annotations

import base64
from pathlib import Path

import pytest
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode
from llama_index.evaluation.nace._drex.errors import NaceImageError, NaceImageLimitError
from llama_index.evaluation.nace._images import split_sources

FIXTURES = Path(__file__).parent / "fixtures"
RED = (FIXTURES / "red.png").read_bytes()
RED_B64 = base64.b64encode(RED).decode()
RED_URL = f"data:image/png;base64,{RED_B64}"


def scored(node: TextNode, score: float | None = 1.0) -> NodeWithScore:
    return NodeWithScore(node=node, score=score)


def test_base64_image_node_to_data_url() -> None:
    node = ImageNode(image=RED_B64, image_mimetype="image/png")
    contexts, images = split_sources([scored(node)], max_images=None)
    assert images == [RED_URL]
    assert contexts == []


def test_image_path_node() -> None:
    node = ImageNode(image_path=str(FIXTURES / "red.png"))
    _, images = split_sources([scored(node)], max_images=None)
    assert images == [RED_URL]


def test_data_url_node() -> None:
    node = ImageNode(image_url=RED_URL)
    _, images = split_sources([scored(node)], max_images=None)
    assert images == [RED_URL]


def test_http_image_url_refused() -> None:
    node = ImageNode(image_url="https://example.com/a.png")
    with pytest.raises(NaceImageError, match="doesn't fetch"):
        split_sources([scored(node)], max_images=None)


def test_image_node_url_refuses_file_paths() -> None:
    node = ImageNode(image_url=str(FIXTURES / "red.png"))
    with pytest.raises(NaceImageError, match="image_paths"):
        split_sources([scored(node)], max_images=None)


def test_image_node_without_source_refused() -> None:
    with pytest.raises(NaceImageError, match="no image"):
        split_sources([scored(ImageNode(text="caption"))], max_images=None)


def test_caption_added_to_contexts() -> None:
    nodes = [
        scored(ImageNode(image=RED_B64, image_mimetype="image/png", text="revenue chart")),
        scored(ImageNode(image=RED_B64, image_mimetype="image/png")),
        scored(ImageNode(image=RED_B64, image_mimetype="image/png", text="second chart")),
    ]
    contexts, images = split_sources(nodes, max_images=None)
    assert contexts == ["[Image 1] revenue chart", "[Image 3] second chart"]
    assert len(images) == 3


def test_text_nodes_become_contexts() -> None:
    nodes = [scored(TextNode(text="alpha")), scored(TextNode(text="beta"))]
    assert split_sources(nodes, max_images=None) == (["alpha", "beta"], [])


def many(count: int) -> list[NodeWithScore]:
    return [
        scored(ImageNode(image=RED_B64, image_mimetype="image/png", text=f"n{i}"), score=i / 10)
        for i in range(count)
    ]


def test_eight_images_pass() -> None:
    _, images = split_sources(many(8), max_images=None)
    assert len(images) == 8


def test_more_than_8_images_needs_max_images() -> None:
    with pytest.raises(NaceImageLimitError, match="max_images"):
        split_sources(many(9), max_images=None)


def test_max_images_keeps_top_scored() -> None:
    nodes = [*many(10), scored(TextNode(text="plain"))]
    contexts, images = split_sources(nodes, max_images=3)
    assert len(images) == 3
    # Highest scores are n9, n8, n7, kept in source order and renumbered.
    assert contexts == ["[Image 1] n7", "[Image 2] n8", "[Image 3] n9", "plain"]


def test_max_images_unscored_nodes_rank_last() -> None:
    nodes = [scored(ImageNode(image=RED_B64, text="none"), score=None), *many(9)]
    contexts, _ = split_sources(nodes, max_images=8)
    assert all("none" not in c for c in contexts)


def test_max_images_applied_under_drex_limit() -> None:
    contexts, images = split_sources([*many(4), scored(TextNode(text="plain"))], max_images=2)
    assert len(images) == 2
    assert contexts == ["[Image 1] n2", "[Image 2] n3", "plain"]


def test_max_images_at_or_above_count_keeps_all() -> None:
    _, images = split_sources(many(4), max_images=4)
    assert len(images) == 4
    _, images = split_sources(many(4), max_images=20)
    assert len(images) == 4
