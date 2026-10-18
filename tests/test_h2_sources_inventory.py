"""H2 (Doc. 58): ferramentas de fontes (formato sintético, nunca conteúdo real), bloqueio de rede documentado e inventário factual do ficheiro local."""
import json
import zipfile
from pathlib import Path

from validation.benchmark import h2_sources as S

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs/evidence/b1b4/h2"

FIXTURE = """SYNTHETIC FORMAT FIXTURE — not a source
Attribute 1: (qualitative)
      Name of the thing
      X1 : first
      X2 : second
Attribute 2: (numerical)
      duration
Attribute 3: (qualitative)
      Y1 : alpha
"""


def test_parser_extracts_blocks_and_codes_from_a_synthetic_format_fixture():
    b = S.parse_attribute_blocks(FIXTURE)
    assert set(b) == {1, 2, 3} and b[1]["codes"] == {"X1": "first", "X2": "second"} and b[2]["codes"] == {} and b[3]["codes"] == {"Y1": "alpha"}
    assert "Name of the thing" in b[1]["label"]


def test_compare_codes_reports_divergences_without_choosing_a_side():
    d = S.compare_codes({1: {"A": "x", "B": "y"}, 2: {"A": "z"}}, {1: {"A": "x2", "C": "w"}, 2: {"A": "z"}})
    assert set(d["divergent_attributes"]) == {"1"} and d["divergent_attributes"]["1"]["only_original"] == ["B"] and d["divergent_attributes"]["1"]["only_corrected"] == ["C"]
    assert d["divergent_attributes"]["1"]["changed"] == {"A": {"original": "x", "corrected": "x2"}}


def test_file_record_hashes_and_marks_content_consulted_only_when_text_was_read(tmp_path):
    t = tmp_path / "fixture.doc"
    t.write_text(FIXTURE, encoding="utf-8")
    r = S.file_record(t, "S1_UCI_STATLOG_GERMAN")
    assert len(r["sha256"]) == 64 and r["content_consulted"] is True and r["n_attribute_blocks"] == 3 and r["origin_verified"] is False
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("codetable.txt", "synthetic")
    rz = S.file_record(z, "S2_UCI_SOUTH_GERMAN")
    assert rz["content_consulted"] is False and "codetable.txt" in rz["zip_members"]
    b = tmp_path / "blob.bin"
    b.write_bytes(b"\x00\x01")
    assert S.file_record(b, "S3_GROEMPING_2019")["content_consulted"] is False


def test_network_block_is_documented_and_no_source_is_claimed_verified():
    p = json.loads((EV / "network_probe.json").read_text())
    assert len(p["attempts"]) == 3 and all("error" in a and "403" in a["error"] for a in p["attempts"])
    m = json.loads((EV / "sources_manifest.json").read_text())
    assert m["status"].startswith("BLOCKED") and m["files"] == [] and m["content_consulted"] is False and m["origin_verified"] is False


def test_german_inventory_is_factual_and_leaves_documentation_pending():
    inv = json.loads((EV / "german_inventory.json").read_text())
    assert inv["n_rows"] == 1000 and inv["n_attribute_columns"] == 20 and inv["n_encoded_features"] == 61 and inv["encoded_columns_total"] == 61
    assert len(inv["attributes"]) == 20 and sum(a["n_encoded_columns"] for a in inv["attributes"]) == 61
    assert all(a["documented_definition"] == "PENDING_SOURCE" and a["documented_ordinal_status"] == "PENDING_SOURCE" and a["south_german_correction"] == "PENDING_SOURCE" for a in inv["attributes"])
    num = [a for a in inv["attributes"] if a["kind_in_project_loader"].startswith("numérico")]
    nom = [a for a in inv["attributes"] if a["kind_in_project_loader"].startswith("nominal")]
    assert len(num) == 7 and len(nom) == 13 and sum(a["n_encoded_columns"] for a in nom) == 54
    assert len(inv["file_sha256"]) == 64
    g = ROOT / "validation/counterfactual_research/datasets/originales/german.data"
    assert S.sha256_file_lf(g) == inv["file_sha256_lf_normalized"]                                                    # o ficheiro histórico não foi alterado
    pri = inv["priority_of_derived_literals"]["rows"]
    assert len(pri) == 10 and sum(r["in_pool_mf12"] for r in pri) == 2                                # só 2 de 10 literais entram no pool do Original por variância (mf=12)


def test_raw_and_canonical_hashes_are_distinct_identifications(tmp_path):
    from validation.benchmark import h2_hash_integrity as HI
    lf, crlf = tmp_path / "a.data", tmp_path / "b.data"
    lf.write_bytes(b"x y\nz w\n")
    crlf.write_bytes(b"x y\r\nz w\r\n")
    a, b = HI.hashes(lf), HI.hashes(crlf)
    assert a["sha256_raw"] != b["sha256_raw"] and a["sha256_canonical"] == b["sha256_canonical"]            # o canónico não troca o bruto
    assert a["raw_equals_canonical"] and not b["raw_equals_canonical"] and b["n_crlf"] == 2 and "CRLF->LF" in b["canonicalization"]
    binf = tmp_path / "c.pdf"
    binf.write_bytes(b"%PDF\r\n\x00")
    h = HI.hashes(binf)
    assert h["canonicalization"] == "none" and h["sha256_raw"] == h["sha256_canonical"]                    # binários não se normalizam
    rec = S.file_record(crlf, "S1_UCI_STATLOG_GERMAN")
    assert rec["sha256_raw"] == b["sha256_raw"] and rec["sha256_canonical"] == b["sha256_canonical"] and rec["origin_verified"] is False


def test_inventory_records_raw_and_canonical_and_raw_checked_only_where_bytes_match():
    inv = json.loads((ROOT / "docs/evidence/b1b4/h2/german_inventory.json").read_text())
    g = ROOT / "validation/counterfactual_research/datasets/originales/german.data"
    assert S.sha256_file_lf(g) == inv["sha256_canonical"] == inv["file_sha256_lf_normalized"]
    if b"\r\n" not in g.read_bytes():                                                                   # checkout LF: o bruto tem de coincidir com o registado
        assert S.sha256_file(g) == inv["sha256_raw"]
