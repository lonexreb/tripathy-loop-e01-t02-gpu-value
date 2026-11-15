from pathlib import Path

from tripathy_arch3.audit import canonical_json, sha256_file
from tripathy_arch3.preflight import verify_pack_manifest


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_pack_manifest_verification_preserves_failures(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("unchanged\n", encoding="utf-8")
    second.write_text("before\n", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(
        f"{sha256_file(first)}  first.txt\n{sha256_file(second)}  second.txt\n",
        encoding="utf-8",
    )

    count, mismatches = verify_pack_manifest(tmp_path)
    assert count == 2
    assert mismatches == []

    second.write_text("after\n", encoding="utf-8")
    count, mismatches = verify_pack_manifest(tmp_path)
    assert count == 2
    assert mismatches == ["second.txt: checksum mismatch"]
