"""
fetch_papers.py
───────────────
Search arXiv and manage the literature registry at docs/literature/registry.json.

Usage:
    # Search arXiv and add results interactively
    python tools/fetch_papers.py search "AWQ quantization vision language model"

    # Add a specific arXiv paper by ID
    python tools/fetch_papers.py add 2306.00978 --tags quantization mobile

    # List all tracked papers
    python tools/fetch_papers.py list

    # List papers with a specific tag
    python tools/fetch_papers.py list --tag quantization
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "docs" / "literature" / "registry.json"
ARXIV_API     = "https://export.arxiv.org/api/query"

VALID_TAGS = [
    "quantization", "pruning", "distillation", "vision-encoder",
    "kv-cache", "mobile", "vlm-arch", "benchmark",
]


# ── Registry helpers ──────────────────────────────────────────────────────────

def load_registry() -> dict:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"papers": []}


def save_registry(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def existing_ids(reg: dict) -> set[str]:
    return {p["id"] for p in reg["papers"]}


# ── arXiv helpers ─────────────────────────────────────────────────────────────

NS = "http://www.w3.org/2005/Atom"

def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall(f"{{{NS}}}entry"):
        arxiv_id = entry.find(f"{{{NS}}}id").text.split("/abs/")[-1].strip()
        title    = entry.find(f"{{{NS}}}title").text.strip().replace("\n", " ")
        abstract = entry.find(f"{{{NS}}}summary").text.strip().replace("\n", " ")
        authors  = [a.find(f"{{{NS}}}name").text
                    for a in entry.findall(f"{{{NS}}}author")]
        published = entry.find(f"{{{NS}}}published").text[:4]  # year only
        papers.append({
            "id": arxiv_id, "title": title, "abstract": abstract,
            "authors": authors, "year": int(published),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return papers


def arxiv_fetch_one(arxiv_id: str) -> dict | None:
    clean_id = arxiv_id.replace("arxiv:", "").strip()
    params = urllib.parse.urlencode({"id_list": clean_id})
    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    entries = root.findall(f"{{{NS}}}entry")
    if not entries:
        return None
    entry = entries[0]
    arxiv_id_out = entry.find(f"{{{NS}}}id").text.split("/abs/")[-1].strip()
    title    = entry.find(f"{{{NS}}}title").text.strip().replace("\n", " ")
    abstract = entry.find(f"{{{NS}}}summary").text.strip().replace("\n", " ")
    authors  = [a.find(f"{{{NS}}}name").text
                for a in entry.findall(f"{{{NS}}}author")]
    published = entry.find(f"{{{NS}}}published").text[:4]
    return {
        "id": arxiv_id_out, "title": title, "abstract": abstract,
        "authors": authors, "year": int(published),
        "url": f"https://arxiv.org/abs/{arxiv_id_out}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id_out}",
    }


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_search(args):
    query = " ".join(args.query)
    print(f"Searching arXiv: '{query}' …")
    results = arxiv_search(query, max_results=args.n)
    reg     = load_registry()
    known   = existing_ids(reg)

    for i, p in enumerate(results):
        already = " [already in registry]" if p["id"] in known else ""
        print(f"\n[{i+1}] {p['id']}{already}")
        print(f"     {p['title']}")
        print(f"     {', '.join(p['authors'][:3])}{'...' if len(p['authors']) > 3 else ''} ({p['year']})")
        print(f"     {p['abstract'][:180]}…")

    if not results:
        print("No results.")
        return

    answer = input("\nAdd which? (comma-separated numbers, or 'none'): ").strip()
    if answer.lower() in ("", "none", "n"):
        return
    indices = [int(x.strip()) - 1 for x in answer.split(",") if x.strip().isdigit()]

    for idx in indices:
        if 0 <= idx < len(results):
            p = results[idx]
            if p["id"] in known:
                print(f"  {p['id']} already in registry, skipping.")
                continue
            tags_in = input(f"  Tags for '{p['id']}' ({', '.join(VALID_TAGS)}): ")
            tags = [t.strip() for t in tags_in.split(",") if t.strip()]
            notes = input(f"  Notes (optional): ").strip()
            entry = {**p, "tags": tags, "notes": notes, "added": str(date.today())}
            reg["papers"].append(entry)
            print(f"  Added: {p['title'][:70]}")

    save_registry(reg)
    print(f"\nRegistry saved → {REGISTRY_PATH}  ({len(reg['papers'])} papers)")


def cmd_add(args):
    arxiv_id = args.arxiv_id.replace("arxiv:", "").strip()
    reg  = load_registry()
    known = existing_ids(reg)
    if arxiv_id in known:
        print(f"{arxiv_id} already in registry.")
        return

    print(f"Fetching {arxiv_id} from arXiv …")
    p = arxiv_fetch_one(arxiv_id)
    if p is None:
        print(f"ERROR: could not fetch {arxiv_id}")
        sys.exit(1)

    tags  = args.tags or []
    notes = args.notes or ""
    entry = {**p, "tags": tags, "notes": notes, "added": str(date.today())}
    reg["papers"].append(entry)
    save_registry(reg)

    print(f"Added: {p['title']}")
    print(f"  Authors: {', '.join(p['authors'][:3])}")
    print(f"  Year:    {p['year']}")
    print(f"  Tags:    {tags}")
    print(f"Registry → {REGISTRY_PATH}  ({len(reg['papers'])} papers)")


def cmd_list(args):
    reg = load_registry()
    papers = reg["papers"]
    if args.tag:
        papers = [p for p in papers if args.tag in p.get("tags", [])]

    if not papers:
        print("No papers found.")
        return

    print(f"{'ID':<20} {'Year':<6} {'Tags':<35} Title")
    print("─" * 100)
    for p in sorted(papers, key=lambda x: x.get("year", 0), reverse=True):
        tags = ", ".join(p.get("tags", []))
        print(f"{p['id']:<20} {p.get('year', '?'):<6} {tags:<35} {p['title'][:50]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Literature registry manager")
    sub = ap.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search", help="Search arXiv and add papers")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-n", type=int, default=8, help="Max results")

    p_add = sub.add_parser("add", help="Add a specific arXiv paper by ID")
    p_add.add_argument("arxiv_id")
    p_add.add_argument("--tags", nargs="+", default=[])
    p_add.add_argument("--notes", default="")

    p_list = sub.add_parser("list", help="List tracked papers")
    p_list.add_argument("--tag", help="Filter by tag")

    args = ap.parse_args()
    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "list":
        cmd_list(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
