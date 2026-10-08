from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _includes(name, seen=None):
    seen = seen or set()
    out = set()
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("-r "):
            sub = line[3:].strip()
            if sub not in seen:
                seen.add(sub)
                out |= {sub} | _includes(sub, seen)
    return out


def test_windows_requirements_cover_gui_and_owl():
    inc = _includes("requirements-win.txt")
    assert {"requirements-gui.txt", "requirements-owl.txt", "requirements-core.txt"} <= inc
