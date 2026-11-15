import pytest

from lungai import pretrained
from lungai.pretrained import download_pretrained, is_downloaded, sha256


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """A local folder served as the release, with matching checksums."""
    release = tmp_path / "release"
    release.mkdir()
    files = {}
    for name, content in [("mobilenetv2_ft.keras", b"weights"), ("mobilenetv2_ft_threshold.json", b"{}")]:
        (release / name).write_bytes(content)
        files[name] = sha256(release / name)
    monkeypatch.setattr(pretrained, "RELEASE_FILES", files)
    monkeypatch.setenv("LUNGAI_RELEASE_URL", release.as_uri())
    return release


def test_download_pretrained(fake_release, tmp_path):
    models = tmp_path / "models"
    assert not is_downloaded(models)

    saved = download_pretrained(models)

    assert sorted(p.name for p in saved) == ["mobilenetv2_ft.keras", "mobilenetv2_ft_threshold.json"]
    assert (models / "mobilenetv2_ft.keras").read_bytes() == b"weights"
    assert is_downloaded(models)
    assert not list(models.glob("*.part"))


def test_download_rejects_bad_checksum(fake_release, tmp_path, monkeypatch):
    monkeypatch.setitem(pretrained.RELEASE_FILES, "mobilenetv2_ft.keras", "0" * 64)
    models = tmp_path / "models"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_pretrained(models)

    assert not (models / "mobilenetv2_ft.keras").exists()
    assert not list(models.glob("*.part"))


def test_release_hashes_are_pinned():
    assert set(pretrained.RELEASE_FILES) == {"mobilenetv2_ft.keras", "mobilenetv2_ft_threshold.json"}
    assert all(len(h) == 64 for h in pretrained.RELEASE_FILES.values())
