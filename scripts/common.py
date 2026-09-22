"""Portable configuration, safe relative inputs and hashed output provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def resolved(path):
    value = str(Path(path).resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


PACKAGE = resolved(__file__).parents[1]


def sha(path):
    with resolved(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root, name):
    root = resolved(root)
    path = resolved(root / name)
    if not path.is_relative_to(root):
        raise ValueError("Configured relative path escapes its data root")
    return path


class Context:
    def __init__(self, args):
        self.root = resolved(args.data_root)
        self.output = resolved(args.output)
        self.config = json.loads(resolved(args.config).read_text(encoding="utf-8"))
        self.reads = {}
        self.output.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        path = safe_path(self.root, self.config["inputs"][key])
        self.reads[path.relative_to(self.root).as_posix()] = sha(path)
        return path

    def csv(self, key):
        return pd.read_csv(self.path(key))

    def json(self, key):
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def write_csv(self, name, data):
        path = safe_path(self.output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)

    def write_json(self, name, value):
        path = safe_path(self.output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def finish(self, script):
        for name, value in self.reads.items():
            if sha(safe_path(self.root, name)) != value:
                raise RuntimeError("Input changed during regeneration")
        self.write_json(
            Path(script).stem + "_bindings.json",
            {
                "script_sha256": sha(script),
                "input_sha256": self.reads,
                "settings": self.config["settings"],
            },
        )


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--data-root", default=str(PACKAGE))
    result.add_argument("--config", default=str(PACKAGE / "config/reproduction.json"))
    result.add_argument("--output", default=str(PACKAGE / "generated"))
    return result
