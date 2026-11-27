"""Create a portable project ZIP, including the measured artifacts and deliverables."""
from pathlib import Path
import json
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "VeriFuse_Project_Final.zip"


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    roots = [ROOT / name for name in ("verifuse", "static", "scripts", "data", "models", "results", "docs", "tests")]
    files = [ROOT / name for name in ("README.md", "app.py", "requirements.txt", "requirements-real.txt", "requirements-docs.txt", "start.ps1", "start.bat", "setup.ps1", ".gitignore")]
    for folder in roots:
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or "build_deliverables_qa" in path.parts or path.suffix in {".pyc", ".part"}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in {"docs/paper.md", "docs/PJ_07_Recovered_Brief.txt"}:
                continue
            # The source mirror does not establish redistribution rights for its media.
            # Download scripts reproduce local data and the pinned large CLIP encoder.
            if relative.startswith(("data/real/", "models/real/clip/")):
                continue
            files.append(path)
    paper_dir = OUTPUT.parent / "IEEE_Verifuse_Paper"
    deliverables = [
        paper_dir / "VeriFuse_IEEE_Final.pdf",
        paper_dir / "main.tex",
        paper_dir / "IEEEtran.cls",
    ]
    missing = [str(path.relative_to(ROOT)) for path in deliverables if not path.is_file()]
    if missing:
        raise SystemExit("Build the final deliverables before packaging: " + ", ".join(missing))
    files.extend(deliverables)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                if relative == "results/real/experiments.json":
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["difficult_cases"] = []
                    value["difficult_case_selection"] = "Downloaded example text is omitted from the distributable archive. Reproduce the local dataset and run to inspect examples."
                    archive.writestr("VeriFuse/" + relative, json.dumps(value, indent=2, ensure_ascii=False))
                else:
                    archive.write(path, "VeriFuse/" + relative)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
