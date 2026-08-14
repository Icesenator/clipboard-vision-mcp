"""Tests unitaires du fallback multi-modèles vision (sans réseau)."""
import asyncio
import base64
import importlib
import io
import os
import tempfile
from pathlib import Path

from PIL import Image

import clipboard_vision_mcp.server as srv


def _reload_server():
    return importlib.reload(srv)


def _png_file() -> Path:
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp = Path(path)
    tmp.write_bytes(buf.getvalue())
    return tmp


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeHTTPXClient:
    def __init__(self, fail_models: list[str], ok_content: str = "REPONSE_OK"):
        self.fail_models = fail_models
        self.ok_content = ok_content
        self.calls: list[str] = []

    async def post(self, url, headers=None, json=None):
        model = json["model"]
        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError("429 rate limited")
        return FakeResponse(
            payload={"choices": [{"message": {"content": self.ok_content}}]}
        )


def test_liste_lue_depuis_env():
    os.environ["GROQ_VISION_MODELS"] = "m1,m2,m3"
    srv2 = _reload_server()
    assert srv2.VISION_MODELS == ["m1", "m2", "m3"]
    del os.environ["GROQ_VISION_MODELS"]


def test_liste_vide_retombe_sur_le_modele_unique():
    os.environ["GROQ_VISION_MODEL"] = "qwen-seul"
    srv2 = _reload_server()
    assert srv2.VISION_MODELS == ["qwen-seul"]
    del os.environ["GROQ_VISION_MODEL"]


def test_fallback_essaie_les_modeles_dans_l_ordre():
    os.environ["GROQ_VISION_MODELS"] = "m1,m2,m3"
    srv2 = _reload_server()
    try:
        client = srv2.VisionClient(api_key="fake")
        fake = FakeHTTPXClient(fail_models=["m1", "m2"])
        client._client = fake
        path = _png_file()
        try:
            result = asyncio.run(client.analyze(str(path), "Decris"))
        finally:
            path.unlink(missing_ok=True)
        assert result == "REPONSE_OK"
        assert fake.calls == ["m1", "m2", "m3"]
    finally:
        del os.environ["GROQ_VISION_MODELS"]


def test_tous_les_modeles_echouent_leve_une_erreur_claire():
    os.environ["GROQ_VISION_MODELS"] = "m1,m2,m3"
    srv2 = _reload_server()
    try:
        client = srv2.VisionClient(api_key="fake")
        fake = FakeHTTPXClient(fail_models=["m1", "m2", "m3"])
        client._client = fake
        path = _png_file()
        try:
            try:
                asyncio.run(client.analyze(str(path), "Decris"))
                raise AssertionError("une erreur etait attendue")
            except RuntimeError as e:
                assert "Vision failed on all models" in str(e)
                assert "m1" in str(e) and "m3" in str(e)
        finally:
            path.unlink(missing_ok=True)
    finally:
        del os.environ["GROQ_VISION_MODELS"]
