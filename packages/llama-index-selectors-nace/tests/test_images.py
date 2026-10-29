import base64
from pathlib import Path

import pytest
from llama_index.selectors.nace._drex.errors import NaceImageError
from llama_index.selectors.nace._drex.images import as_data_url, to_drex_image

FIXTURES = Path(__file__).parent / "fixtures"


def test_png_bytes_to_base64() -> None:
    data = (FIXTURES / "red.png").read_bytes()
    out = to_drex_image(data)
    assert isinstance(out, str)
    header, _, payload = out.partition(",")
    assert header == "data:image/png;base64"
    assert base64.b64decode(payload) == data


def test_data_url_passthrough() -> None:
    url = "data:image/png;base64,AAAA"
    assert to_drex_image(url) == url


def test_http_url_refused() -> None:
    with pytest.raises(NaceImageError, match="doesn't fetch"):
        to_drex_image("https://example.com/a.png")


def test_missing_path_refused(tmp_path: Path) -> None:
    with pytest.raises(NaceImageError):
        to_drex_image(tmp_path / "nope.png")
    with pytest.raises(NaceImageError):
        to_drex_image(str(tmp_path / "nope.png"))


def test_existing_path_and_str_path() -> None:
    path = FIXTURES / "blue.png"
    expected = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    assert to_drex_image(path) == expected
    assert to_drex_image(str(path)) == expected


def test_workspace_file_passthrough() -> None:
    block = {"type": "workspace_file", "path": "a/b.png"}
    assert to_drex_image(block) == block


def test_unsupported_inputs_refused() -> None:
    with pytest.raises(NaceImageError):
        to_drex_image({"type": "image_url", "image_url": "https://x/y.png"})
    with pytest.raises(NaceImageError):
        to_drex_image(42)  # type: ignore[arg-type]


def test_as_data_url_detects_png_jpeg_webp() -> None:
    def enc(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    png = (FIXTURES / "red.png").read_bytes()
    assert as_data_url(enc(png)).startswith("data:image/png;base64,")
    assert as_data_url(enc(b"\xff\xd8\xff\xe0" + b"0" * 20)).startswith("data:image/jpeg;base64,")
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"0" * 8
    assert as_data_url(enc(webp)).startswith("data:image/webp;base64,")
    assert as_data_url(enc(b"GIF89a" + b"0" * 20)).startswith("data:image/gif;base64,")
    assert as_data_url(enc(b"unknown bytes here!!")).startswith(
        "data:application/octet-stream;base64,"
    )
    assert as_data_url("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


def test_existing_relative_path_that_is_valid_base64_is_read_as_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = (FIXTURES / "blue.png").read_bytes()
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "blue").write_bytes(data)
    monkeypatch.chdir(tmp_path)
    base64.b64decode("img/blue", validate=True)  # the name itself is valid base64
    assert to_drex_image("img/blue") == to_drex_image(data)
