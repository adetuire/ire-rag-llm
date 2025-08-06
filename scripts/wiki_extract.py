#!/usr/bin/env python
"""
Extract plain-text pages from a Wikipedia XML dump (offline).
Usage:
  python scripts/wiki_extract.py --dump <dump.xml.bz2> --out data/wiki_docs --limit 20000
"""
import argparse, bz2, re, sys
from pathlib import Path
import mwxml
import mwparserfromhell as mw

def clean_title(title: str) -> str:
    title = re.sub(r"[^\w\s\-]", "-", title)
    return re.sub(r"\s+", "_", title)[:80]

def iter_pages(dump_path: Path):
    with bz2.open(dump_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            # namespace 0 = main/article space
            ns = getattr(page, "namespace", 0)
            if ns != 0:
                continue
            latest_text = None
            for rev in page:  # stream revisions; keep last
                latest_text = rev.text
            yield page.title, (latest_text or "")

def to_plain_text(wikitext: str) -> str:
    return mw.parse(wikitext).strip_code().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Path to *-pages-articles.xml.bz2")
    ap.add_argument("--out", required=True, help="Output folder for .txt files")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N pages")
    args = ap.parse_args()

    dump_path = Path(args.dump)
    if not dump_path.exists():
        print(f"[error] dump not found: {dump_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for title, raw in iter_pages(dump_path):
        if not raw:
            continue
        plain = to_plain_text(raw)
        if len(plain) < 200:  # skip tiny stubs
            continue
        fname = out_dir / f"{count:08d}_{clean_title(title)}.txt"
        fname.write_text(plain, encoding="utf-8")
        count += 1
        if count % 2000 == 0:
            print(f"saved {count} pages...")
        if args.limit and count >= args.limit:
            break

    print(f"Done. Saved {count} pages to {out_dir.resolve()}")

if __name__ == "__main__":
    main()

# This script extracts plain-text pages from a Wikipedia XML dump.
# It processes the dump, cleans titles, converts wikitext to plain text,
# and saves the results as individual .txt files.