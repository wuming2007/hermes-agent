import base64
import json
import sys
import types

import pytest

sys.modules.setdefault("fal_client", types.SimpleNamespace())

from tools import image_generation_tool as imgtool


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_provider_prefers_gemini_when_google_key_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_IMAGE_PROVIDER", raising=False)

    assert imgtool._get_image_generation_provider() == "gemini"


def test_provider_can_force_fal(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "fal")

    assert imgtool._get_image_generation_provider() == "fal"


def test_gemini_generation_maps_aspect_ratio_and_saves_base64(monkeypatch, tmp_path):
    calls = []
    png_bytes = b"fake-png-bytes"

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_IMAGE_PROVIDER", raising=False)

    def fake_post(url, params, json, timeout):
        calls.append({"url": url, "params": params, "json": json, "timeout": timeout})
        return DummyResponse({
            "predictions": [{
                "bytesBase64Encoded": base64.b64encode(png_bytes).decode("ascii"),
                "mimeType": "image/png",
            }]
        })

    monkeypatch.setattr(imgtool.requests, "post", fake_post)

    result = json.loads(imgtool.image_generate_tool("draw a seed", aspect_ratio="portrait"))

    assert result["success"] is True
    assert result["image"].startswith(str(tmp_path / "generated-images"))
    assert (tmp_path / "generated-images").exists()
    assert calls[0]["json"]["parameters"]["aspectRatio"] == "9:16"
    assert calls[0]["json"]["parameters"]["sampleCount"] == 1
    assert calls[0]["params"] == {"key": "test-key"}


def test_gemini_generation_uses_image_url_without_saving(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fake_post(url, params, json, timeout):
        return DummyResponse({"predictions": [{"imageUrl": "https://example.test/image.png"}]})

    monkeypatch.setattr(imgtool.requests, "post", fake_post)

    result = json.loads(imgtool.image_generate_tool("draw a seed", aspect_ratio="landscape"))

    assert result == {"success": True, "image": "https://example.test/image.png"}
    assert not (tmp_path / "generated-images").exists()


def test_fal_fallback_when_no_gemini_key(monkeypatch):
    calls = []

    class Handler:
        def get(self):
            return {"images": [{"url": "https://fal.test/original.png", "width": 10, "height": 10}]}

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_IMAGE_PROVIDER", raising=False)
    monkeypatch.setenv("FAL_KEY", "fal-key")
    monkeypatch.setattr(imgtool, "_resolve_managed_fal_gateway", lambda: None)
    monkeypatch.setattr(imgtool, "_upscale_image", lambda url, prompt: None)

    def fake_submit(model, arguments):
        calls.append({"model": model, "arguments": arguments})
        return Handler()

    monkeypatch.setattr(imgtool, "_submit_fal_request", fake_submit)

    result = json.loads(imgtool.image_generate_tool("draw a seed", aspect_ratio="square"))

    assert result == {"success": True, "image": "https://fal.test/original.png"}
    assert calls[0]["model"] == imgtool.DEFAULT_MODEL
    assert calls[0]["arguments"]["image_size"] == "square_hd"


def test_requirements_true_with_google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("FAL_KEY", raising=False)

    assert imgtool.check_image_generation_requirements() is True
