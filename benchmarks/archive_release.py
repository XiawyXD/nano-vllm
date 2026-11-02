"""把一次已结束实验的原始证据打包并校验；不删除原文件，不包含模型权重。"""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--standalone", action="store_true",
                        help="archive a completed operator/CPU experiment with no HTTP service session")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    directory = Path(args.directory).resolve()
    if not directory.is_relative_to(repo / "artifacts") or not directory.is_dir():
        raise ValueError("archive directory must be an existing directory below project artifacts")
    session_path = directory / "service_session/session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text())
        if session.get("status") == "RUNNING":
            raise ValueError("refusing to archive a running session")
    elif not args.standalone:
        raise ValueError("service session missing; use --standalone only for completed non-service experiments")
    if args.standalone:
        for path in directory.rglob("*.json"):
            report = json.loads(path.read_text())
            if isinstance(report, dict) and report.get("status") == "RUNNING":
                raise ValueError(f"refusing to archive unfinished experiment: {path}")
    excluded = {"raw-data.tar.gz", "manifest.json", ".gitignore", "report.md"}
    files = sorted(path for path in directory.rglob("*")
                   if path.is_file() and not path.is_symlink()
                   and path.name not in excluded)
    entries = [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                "sha256": digest(path)} for path in files]
    archive = directory / "raw-data.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=5) as target:
        for path in files:
            target.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    # 解压后逐文件hash核对，确保归档没有遗漏/损坏，避免只检查压缩文件存在。
    with tarfile.open(archive, "r:gz") as packed:
        for item in entries:
            with packed.extractfile(item["path"]) as data:
                value = hashlib.sha256(data.read()).hexdigest()
            assert value == item["sha256"], item["path"]
    manifest = {"source_head_at_archive": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "archive": archive.name, "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest(archive), "files": entries,
                "verification": "Every archived file re-read and SHA256 matched; originals retained",
                "extract": f"tar -xzf {archive.name} -C ."}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"files": len(entries), "archive_bytes": manifest["archive_bytes"],
                      "sha256": manifest["archive_sha256"]}))


if __name__ == "__main__":
    main()
