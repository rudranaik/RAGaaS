# rag_mvp/loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict

# ---- Public data shape ------------------------------------------------------

@dataclass
class RawDoc:
    source_id: str          # stable id (e.g., relative path)
    text: str               # full extracted text
    metadata: Dict[str, str]  # filename, ext, size, etc.

# ---- Core API ---------------------------------------------------------------

def load_folder(
    root: str | Path = "data/raw",
    include_ext: Optional[List[str]] = None,
) -> List[RawDoc]:
    """
    Walk 'root' and load files into RawDoc records.
    Default includes: .txt, .md
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if include_ext is None:
        include_ext = [".txt", ".md"]  # start simple; we’ll add more later

    docs: List[RawDoc] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_ext:
            continue

        doc = _load_one(path, root)
        if doc:
            docs.append(doc)
    return docs

# ---- File dispatch ----------------------------------------------------------

def _load_one(path: Path, root: Path) -> Optional[RawDoc]:
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return _load_text_like(path, root)

    # Stubs we’ll fill later:
    if ext == ".pdf":
        # return _load_pdf(path, root)
        return None
    if ext in {".mp3", ".wav", ".m4a"}:
        # return _load_audio(path, root)
        return None
    if ext in {".mp4", ".mov", ".mkv"}:
        # return _load_video(path, root)
        return None

    return None

# ---- Extractors -------------------------------------------------------------

def _load_text_like(path: Path, root: Path) -> RawDoc:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return RawDoc(
        source_id=str(path.relative_to(root)),
        text=_normalize_newlines(text),
        metadata={
            "filename": path.name,
            "ext": path.suffix.lower(),
            "bytes": str(path.stat().st_size),
            "relpath": str(path.relative_to(root)),
            "abspath": str(path.resolve()),
        },
    )

# ---- Utilities --------------------------------------------------------------

def _normalize_newlines(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines to 2
    out: List[str] = []
    blank_run = 0
    for line in s.split("\n"):
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip()
