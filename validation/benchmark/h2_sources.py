"""H2 (Doc. 58): ferramentas para registar e consultar FONTES DOCUMENTAIS do German Credit sem inventar conteúdo.
* ``probe`` regista o estado do acesso de rede aos anfitriões das fontes (sem contornar políticas).
* ``register`` calcula SHA-256 e tamanho de ficheiros FORNECIDOS (ou descarregados) e só marca ``content_consulted = true`` se o texto foi efetivamente extraído e analisado.
* ``parse_attribute_blocks`` extrai, de forma tolerante, blocos «Attribute N: … código : descrição» de um texto ``german.doc``; os testes usam um *fixture* SINTÉTICO só para validar o formato do analisador,
  nunca como conteúdo da fonte. ``compare_codes`` compara duas tabelas normalizadas (original vs corrigida).
Nada aqui afirma o conteúdo de uma fonte: sem ficheiro, tudo fica ``PENDING_SOURCE``. A origem (URL) é a DECLARADA pelo responsável; os repositórios não publicam hashes, pelo que a origem não é verificável por máquina."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SOURCES = {
    "S1_UCI_STATLOG_GERMAN": {"declared_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc", "declared_url_alt": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data", "expected": ["german.doc", "german.data"], "role": "documentação original (códigos clássicos)"},
    "S2_UCI_SOUTH_GERMAN": {"declared_url": "https://archive.ics.uci.edu/dataset/573/south+german+credit+update", "declared_url_alt": "https://archive.ics.uci.edu/dataset/522/south+german+credit", "expected": ["codetable / SouthGermanCredit.*"], "role": "versão com tabela de códigos corrigida"},
    "S3_GROEMPING_2019": {"declared_url": "https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf", "expected": ["Report-2019-004.pdf"], "role": "relatório (Grömping, 2019) sobre erros na tabela de códigos clássica"},
}


def sha256_file_lf(p) -> str:
    """SHA-256 após normalizar CRLF→LF (estável entre Linux e Windows com ``autocrlf``); o hash bruto continua em ``sha256_file``."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(p: Path) -> Optional[str]:
    """Texto do ficheiro, ou None se não for possível extrair (PDF sem ferramenta, binário…). Nunca devolve texto inventado."""
    p = Path(p)
    suf = p.suffix.lower()
    try:
        if suf in (".doc", ".txt", ".asc", ".md", ".csv", ".r"):
            return p.read_text(encoding="utf-8", errors="replace")
        if suf == ".pdf":
            try:
                import pypdf  # type: ignore
                return "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(str(p)).pages)
            except Exception:
                r = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, timeout=120)
                return r.stdout if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None
    return None


ATTR_RE = re.compile(r"^\s*Attribute\s+(\d+)\s*:\s*(.*)$", re.I)
CODE_RE = re.compile(r"^\s*([A-Za-z]\d{1,3}|\d{1,3})\s*:\s*(.+?)\s*$")


def parse_attribute_blocks(text: str) -> Dict[int, Dict[str, Any]]:
    """{n: {"header": str, "label": str, "codes": {código: descrição}}}. Tolerante: uma linha «Attribute N:» abre um bloco; as linhas «código : descrição» seguintes pertencem-lhe; linhas de texto livre
    entre o cabeçalho e o primeiro código formam o rótulo. Não interpreta semântica (ordem, tipo)."""
    out: Dict[int, Dict[str, Any]] = {}
    cur: Optional[int] = None
    for line in text.splitlines():
        m = ATTR_RE.match(line)
        if m:
            cur = int(m.group(1))
            out[cur] = {"header": m.group(2).strip(), "label": "", "codes": {}}
            continue
        if cur is None:
            continue
        c = CODE_RE.match(line)
        if c and len(c.group(1)) <= 4:
            out[cur]["codes"][c.group(1)] = c.group(2)
        elif line.strip() and not out[cur]["codes"]:
            out[cur]["label"] = (out[cur]["label"] + " " + line.strip()).strip()
    return out


def compare_codes(original: Dict[Any, Dict[str, str]], corrected: Dict[Any, Dict[str, str]]) -> Dict[str, Any]:
    """Compara tabelas normalizadas {atributo: {código: descrição}}; devolve divergências por atributo e código (sem juízo de qual é a correta)."""
    diffs: Dict[str, Any] = {}
    for a in sorted(set(original) | set(corrected), key=str):
        o, c = original.get(a, {}), corrected.get(a, {})
        d = {"only_original": sorted(set(o) - set(c)), "only_corrected": sorted(set(c) - set(o)), "changed": {k: {"original": o[k], "corrected": c[k]} for k in set(o) & set(c) if o[k] != c[k]}}
        if any(d.values()):
            diffs[str(a)] = d
    return {"n_attributes_compared": len(set(original) | set(corrected)), "divergent_attributes": diffs}


def file_record(path, source_id: str, label: str = "") -> Dict[str, Any]:
    p = Path(path)
    rec: Dict[str, Any] = {"source_id": source_id, "label": label or p.name, "declared_url": SOURCES[source_id]["declared_url"], "file": p.name, "sha256": sha256_file(p), "sha256_raw": sha256_file(p), "sha256_canonical": sha256_file_lf(p) if p.suffix.lower() in {".doc", ".txt", ".csv", ".data", ".arff", ".json", ".md", ".xml"} else sha256_file(p),
                           "canonicalization": "CRLF->LF (apenas)" if p.suffix.lower() in {".doc", ".txt", ".csv", ".data", ".arff", ".json", ".md", ".xml"} else "none", "size_bytes": p.stat().st_size,
                           "origin_verified": False, "origin_note": "origem DECLARADA pelo responsável; o hash local não prova a origem (os repositórios não publicam hashes)", "content_consulted": False, "n_attribute_blocks": 0}
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            rec["zip_members"] = {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.is_dir()}
        rec["content_consulted"] = False
        rec["note"] = "arquivo listado e cada membro com hash; o conteúdo só conta como consultado depois de extraído e analisado"
        return rec
    text = extract_text(p)
    if text is not None and text.strip():
        rec["content_consulted"] = True
        rec["n_chars"] = len(text)
        blocks = parse_attribute_blocks(text)
        rec["n_attribute_blocks"] = len(blocks)
    return rec


def probe(out: Optional[str] = None) -> Dict[str, Any]:
    """Regista o resultado do acesso de rede (sem contornar políticas): código HTTP de uma tentativa por anfitrião."""
    import urllib.request
    res = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "attempts": []}
    for sid, s in SOURCES.items():
        try:
            with urllib.request.urlopen(s["declared_url"], timeout=25) as r:
                res["attempts"].append({"source_id": sid, "url": s["declared_url"], "status": r.status})
        except Exception as e:                                                                  # política de rede, DNS, etc.
            res["attempts"].append({"source_id": sid, "url": s["declared_url"], "error": type(e).__name__ + ": " + str(e)[:200]})
    if out:
        Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main(argv):
    cmd = argv[1]
    if cmd == "probe":
        print(json.dumps(probe(argv[2] if len(argv) > 2 else None), indent=1))
    elif cmd == "register":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--german-doc"); ap.add_argument("--south-file", action="append", default=[]); ap.add_argument("--report-pdf"); ap.add_argument("--out", required=True)
        a = ap.parse_args(argv[2:])
        recs = []
        if a.german_doc:
            recs.append(file_record(a.german_doc, "S1_UCI_STATLOG_GERMAN"))
        for f in a.south_file:
            recs.append(file_record(f, "S2_UCI_SOUTH_GERMAN"))
        if a.report_pdf:
            recs.append(file_record(a.report_pdf, "S3_GROEMPING_2019"))
        Path(a.out).write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "sources": SOURCES, "files": recs}, indent=1, ensure_ascii=False))
        print(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main(sys.argv)
