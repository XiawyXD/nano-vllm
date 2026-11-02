"""Verify/extract official-wheel subset and probe native registration without GPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

BUNDLE_SHA256 = "31bddcda62bf88266359fdff66718daefd683450567f3f5724f300eae26c8d35"
WHEEL_SHA256 = "c52e892309532b4e51cb94d022c5e3c0087300cdb56e4645708601443299d871"
SO_SHA256 = "10ec98d19907ec8d31cc4425064c41da4a704789196020bd3c0f36de581d82b9"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.bundle) != BUNDLE_SHA256:
        raise ValueError("sidecar transport bundle checksum mismatch")
    args.directory.mkdir(exist_ok=False)
    with zipfile.ZipFile(args.bundle) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe archive member")
        archive.extractall(args.directory)
    manifest = json.loads((args.directory / "sidecar_manifest.json").read_text())
    if manifest["wheel_sha256"] != WHEEL_SHA256:
        raise ValueError("unapproved wheel")
    record_file = args.directory / "vllm-0.17.1.dist-info/RECORD"
    with record_file.open(newline="") as file:
        record = {row[0]: row[1:] for row in csv.reader(file)}
    for item in manifest["files"]:
        path = args.directory / item["path"]
        actual = digest(path)
        if actual != item["sha256"] or path.stat().st_size != item["size"]:
            raise ValueError("extracted file digest mismatch")
        if item["path"].endswith("/RECORD"):
            continue
        wheel_hash, size = record[item["path"]]
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
        if wheel_hash != "sha256=" + encoded or int(size) != item["size"]:
            raise ValueError("wheel RECORD mismatch")
    so = args.directory / "vllm/_C.abi3.so"
    if digest(so) != SO_SHA256:
        raise ValueError("unapproved shared object")
    report = dict(wheel_sha256=WHEEL_SHA256, so_sha256=SO_SHA256,
                  native_library=str(so.absolute()), official_record_verified=True,
                  elf_dynamic=subprocess.check_output(["readelf", "-d", str(so)], text=True),
                  elf_versions=subprocess.check_output(["readelf", "--version-info", str(so)], text=True))
    import torch
    report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                  cuda_initialized_before=torch.cuda.is_initialized())
    try:
        torch.ops.load_library(str(so.absolute()))
        report["schemas"] = {name: str(getattr(torch.ops._C, name).default._schema)
                             for name in ("awq_marlin_repack", "marlin_gemm")}
        report["status"] = "CPU_DLOPEN_AND_SCHEMA_PASS_GPU_NOT_RUN"
    except Exception as error:
        report.update(status="DLOPEN_FAILED", error=repr(error))
    report["cuda_initialized_after"] = torch.cuda.is_initialized()
    report["mapped_cuda_torch_libraries"] = sorted(set(line.split()[-1] for line in
        Path("/proc/self/maps").read_text().splitlines()
        if any(s in line for s in ("libcudart", "libtorch", "libc10", "libnvJitLink"))))
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("elf_")}, indent=2))


if __name__ == "__main__":
    main()
