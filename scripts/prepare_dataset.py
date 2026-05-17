"""Converte datasets de detecção para o formato COCO JSON interno.

Detecta automaticamente o formato de entrada (COCO/YOLO/VOC), converte e
escreve em ``data/processed/`` no layout:

  data/processed/
    train/images/   train/annotations.json
    val/images/     val/annotations.json
    test/images/    test/annotations.json   (se existir no input)

Se não houver split de validação no input, cria 80/20 automaticamente a
partir do treino.

Uso:
    python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --format auto
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shared.utils import ensure_dir, get_logger, set_global_seed  # noqa: E402

_log = get_logger("prepare_dataset")

Format = Literal["coco", "yolo", "voc", "auto"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────────────────────────
# Detecção de formato
# ──────────────────────────────────────────────────────────────────


def detect_format(input_root: Path) -> Format:
    """Detecta o formato do dataset inspecionando arquivos."""
    if any(input_root.rglob("*.json")):
        for j in input_root.rglob("*.json"):
            try:
                with j.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if all(k in data for k in ("images", "annotations", "categories")):
                    return "coco"
            except (json.JSONDecodeError, OSError):
                continue
    if any(input_root.rglob("*.xml")):
        for x in input_root.rglob("*.xml"):
            try:
                root = ET.parse(x).getroot()
                if root.tag == "annotation" and root.find("object") is not None:
                    return "voc"
            except ET.ParseError:
                continue
    if any(input_root.rglob("*.txt")):
        # YOLO: arquivos .txt com 5 colunas (class cx cy w h).
        for t in input_root.rglob("*.txt"):
            try:
                with t.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if not first:
                    continue
                parts = first.split()
                if len(parts) == 5 and all(_is_number(p) for p in parts[1:]):
                    return "yolo"
            except OSError:
                continue
    raise RuntimeError(
        f"Não foi possível detectar formato em {input_root}. Use --format coco|yolo|voc."
    )


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


@dataclass
class SplitData:
    images: list[dict]
    annotations: list[dict]
    image_files: dict[int, Path]  # image_id → caminho original


def _make_coco(split: SplitData, categories: list[dict]) -> dict:
    return {
        "info": {"description": "Convertido por prepare_dataset.py"},
        "licenses": [],
        "images": split.images,
        "annotations": split.annotations,
        "categories": categories,
    }


def _copy_split(split: SplitData, dest_root: Path, categories: list[dict]) -> int:
    img_dir = ensure_dir(dest_root / "images")
    for img_id, src in split.image_files.items():
        dst = img_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        # Garante que o file_name no JSON é só o nome (sem subdir).
        for img in split.images:
            if img["id"] == img_id:
                img["file_name"] = src.name
                break
    with (dest_root / "annotations.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_coco(split, categories), fh, indent=2)
    return len(split.images)


def _find_split_dir(input_root: Path, name: str) -> Optional[Path]:
    for candidate in (
        input_root / name,
        input_root / "images" / name,
        input_root / name / "images",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


# ──────────────────────────────────────────────────────────────────
# Converters
# ──────────────────────────────────────────────────────────────────


def _convert_coco(input_root: Path) -> dict[str, SplitData]:
    """Lê COCO JSONs já existentes.

    Suporta dois layouts:
      A) Multi-JSON  — train.json, val.json, test.json separados.
      B) Single-JSON — 1 JSON + pastas físicas train/, val/, test/ com imagens.
         O split é determinado pela pasta física onde cada imagem mora.
    """
    valid_jsons: list[tuple[Path, dict]] = []
    for j in input_root.rglob("*.json"):
        try:
            with j.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if all(k in data for k in ("images", "annotations", "categories")):
            valid_jsons.append((j, data))

    if not valid_jsons:
        raise RuntimeError("Nenhum JSON COCO válido encontrado em " + str(input_root))

    # Layout B: 1 JSON único + pastas físicas com imagens → split por pasta.
    physical_dirs = {
        name: input_root / name
        for name in ("train", "val", "test")
        if (input_root / name).is_dir() and _list_images(input_root / name)
    }
    only_one_json = len(valid_jsons) == 1
    json_has_split_in_name = any(
        s in valid_jsons[0][0].stem.lower() for s in ("train", "val", "test")
    )
    if only_one_json and len(physical_dirs) >= 2 and not json_has_split_in_name:
        return _convert_coco_single_json_physical_split(valid_jsons[0], physical_dirs)

    # Layout A: cada JSON descreve seu split.
    splits: dict[str, SplitData] = {}
    for j, data in valid_jsons:
        name = j.stem.lower()
        if "train" in name:
            split_name = "train"
        elif "val" in name:
            split_name = "val"
        elif "test" in name:
            split_name = "test"
        else:
            split_name = "train"

        img_dir = j.parent / "images"
        if not img_dir.is_dir():
            img_dir = j.parent

        image_files = {}
        for img in data["images"]:
            candidate = img_dir / img["file_name"]
            if not candidate.exists():
                candidate = next((p for p in input_root.rglob(Path(img["file_name"]).name)), candidate)
            image_files[img["id"]] = candidate

        splits[split_name] = SplitData(
            images=data["images"],
            annotations=data["annotations"],
            image_files=image_files,
        )
        splits.setdefault("_categories", data["categories"])  # type: ignore

    return splits


def _convert_coco_single_json_physical_split(
    json_entry: tuple[Path, dict],
    physical_dirs: dict[str, Path],
) -> dict[str, SplitData]:
    """1 JSON único + pastas físicas train/val/test → splita por pasta.

    Mapeia cada imagem do JSON (por basename) para a pasta física onde
    ela mora. Anotações seguem a imagem correspondente.
    """
    json_path, data = json_entry
    _log.info(
        "Detectado layout 'single-JSON + pastas físicas': %s + %s",
        json_path.name, sorted(physical_dirs.keys()),
    )

    # Indexa todas as imagens físicas por basename.
    basename_to_split: dict[str, str] = {}
    basename_to_path: dict[str, Path] = {}
    for split_name, dir_path in physical_dirs.items():
        for img_path in _list_images(dir_path):
            basename_to_split[img_path.name] = split_name
            basename_to_path[img_path.name] = img_path

    # Distribui imagens do JSON pelos splits.
    split_buckets: dict[str, dict] = {
        name: {"images": [], "annotations": [], "image_files": {}, "ids": set()}
        for name in physical_dirs
    }
    unmatched = 0
    for img in data["images"]:
        basename = Path(img["file_name"]).name
        split = basename_to_split.get(basename)
        if split is None:
            unmatched += 1
            continue
        bucket = split_buckets[split]
        # Normaliza file_name para basename (será copiada plana pra processed/).
        img_normalized = dict(img)
        img_normalized["file_name"] = basename
        bucket["images"].append(img_normalized)
        bucket["image_files"][img["id"]] = basename_to_path[basename]
        bucket["ids"].add(img["id"])

    if unmatched:
        _log.warning("%d imagens do JSON não foram encontradas nas pastas físicas.", unmatched)

    # Distribui anotações.
    for ann in data["annotations"]:
        for bucket in split_buckets.values():
            if ann["image_id"] in bucket["ids"]:
                bucket["annotations"].append(ann)
                break

    splits: dict[str, SplitData] = {}
    for name, b in split_buckets.items():
        if not b["images"]:
            continue
        splits[name] = SplitData(
            images=b["images"], annotations=b["annotations"], image_files=b["image_files"]
        )
        _log.info("  %s: %d imagens, %d anotações", name, len(b["images"]), len(b["annotations"]))

    splits["_categories"] = data["categories"]  # type: ignore
    return splits


def _convert_yolo(input_root: Path) -> dict[str, SplitData]:
    """Converte YOLO TXT (um .txt por imagem) para COCO."""
    # classes.txt ou data.yaml
    class_names: list[str] = []
    classes_txt = next(iter(input_root.rglob("classes.txt")), None)
    if classes_txt:
        class_names = [
            line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        data_yaml = next(iter(input_root.rglob("data.yaml")), None)
        if data_yaml:
            import yaml

            cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            names = cfg.get("names")
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names)]
            elif isinstance(names, list):
                class_names = names

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)] if class_names else []
    splits: dict[str, SplitData] = {}

    for split_name in ("train", "val", "test"):
        split_dir = _find_split_dir(input_root, split_name)
        if not split_dir:
            continue
        images, annotations, files = [], [], {}
        ann_id = 1
        for img_id, img_path in enumerate(_list_images(split_dir), start=1):
            with Image.open(img_path) as im:
                w, h = im.size
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            files[img_id] = img_path
            # Procura .txt correspondente em labels/ irmão ou mesmo dir.
            txt = img_path.with_suffix(".txt")
            if not txt.exists():
                txt = split_dir.parent / "labels" / split_name / img_path.with_suffix(".txt").name
            if not txt.exists():
                continue
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = parts
                cls_i = int(cls)
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x, y = cx - bw / 2, cy - bh / 2
                if not categories:
                    pass  # vamos preencher abaixo
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_i + 1,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        splits[split_name] = SplitData(images=images, annotations=annotations, image_files=files)

    # Se classes não foram encontradas, infere a partir das anotações.
    if not categories and splits:
        max_cls = 0
        for s in splits.values():
            for a in s.annotations:
                max_cls = max(max_cls, a["category_id"])
        categories = [{"id": i, "name": f"class_{i - 1}"} for i in range(1, max_cls + 1)]
    splits["_categories"] = categories  # type: ignore
    return splits


def _convert_voc(input_root: Path) -> dict[str, SplitData]:
    """Converte Pascal VOC (um .xml por imagem) para COCO."""
    class_to_id: dict[str, int] = {}
    splits: dict[str, SplitData] = {}

    for split_name in ("train", "val", "test"):
        split_dir = _find_split_dir(input_root, split_name)
        if not split_dir:
            continue
        images, annotations, files = [], [], {}
        ann_id = 1
        for img_id, img_path in enumerate(_list_images(split_dir), start=1):
            xml_candidates = [
                img_path.with_suffix(".xml"),
                split_dir.parent / "Annotations" / img_path.with_suffix(".xml").name,
                split_dir / "Annotations" / img_path.with_suffix(".xml").name,
            ]
            xml = next((x for x in xml_candidates if x.exists()), None)
            with Image.open(img_path) as im:
                w, h = im.size
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            files[img_id] = img_path
            if not xml:
                continue
            root = ET.parse(xml).getroot()
            for obj in root.findall("object"):
                name = obj.findtext("name", "unknown")
                if name not in class_to_id:
                    class_to_id[name] = len(class_to_id) + 1
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                x1 = float(bb.findtext("xmin", 0))
                y1 = float(bb.findtext("ymin", 0))
                x2 = float(bb.findtext("xmax", 0))
                y2 = float(bb.findtext("ymax", 0))
                bw, bh = x2 - x1, y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_to_id[name],
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        splits[split_name] = SplitData(images=images, annotations=annotations, image_files=files)

    categories = [{"id": v, "name": k} for k, v in sorted(class_to_id.items(), key=lambda kv: kv[1])]
    splits["_categories"] = categories  # type: ignore
    return splits


# ──────────────────────────────────────────────────────────────────
# Split automático train→val (80/20) se val ausente
# ──────────────────────────────────────────────────────────────────


def _autosplit_val(train: SplitData, ratio: float = 0.2, seed: int = 42) -> tuple[SplitData, SplitData]:
    rng = random.Random(seed)
    img_ids = [img["id"] for img in train.images]
    rng.shuffle(img_ids)
    k = max(1, int(len(img_ids) * ratio))
    val_ids = set(img_ids[:k])

    def split_for(ids: set[int]) -> SplitData:
        imgs = [i for i in train.images if i["id"] in ids]
        anns = [a for a in train.annotations if a["image_id"] in ids]
        files = {iid: p for iid, p in train.image_files.items() if iid in ids}
        return SplitData(images=imgs, annotations=anns, image_files=files)

    val = split_for(val_ids)
    new_train = split_for(set(img_ids[k:]))
    return new_train, val


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Converte dataset para COCO JSON.")
    p.add_argument("--input", type=Path, required=True, help="Pasta do dataset bruto.")
    p.add_argument("--output", type=Path, required=True, help="Pasta de saída (processed/).")
    p.add_argument("--format", type=str, default="auto", choices=["auto", "coco", "yolo", "voc"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fração do train→val quando val ausente.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    fmt = args.format if args.format != "auto" else detect_format(args.input)
    _log.info("Formato: %s", fmt)

    if fmt == "coco":
        splits = _convert_coco(args.input)
    elif fmt == "yolo":
        splits = _convert_yolo(args.input)
    elif fmt == "voc":
        splits = _convert_voc(args.input)
    else:
        raise ValueError(f"Formato desconhecido: {fmt}")

    categories = splits.pop("_categories")  # type: ignore
    train = splits.get("train")
    if not train:
        raise RuntimeError("Split de treino não encontrado.")
    val = splits.get("val")
    test = splits.get("test")

    if not val:
        _log.warning("Split val ausente — criando 80/20 a partir do treino (seed=%d).", args.seed)
        train, val = _autosplit_val(train, ratio=args.val_ratio, seed=args.seed)

    ensure_dir(args.output)
    n_train = _copy_split(train, ensure_dir(args.output / "train"), categories)
    n_val = _copy_split(val, ensure_dir(args.output / "val"), categories)
    n_test = _copy_split(test, ensure_dir(args.output / "test"), categories) if test else 0

    # Resumo.
    print("\n" + "=" * 60)
    print(f"Dataset convertido em {args.output}")
    print("-" * 60)
    print(f"  treino:    {n_train} imagens, {len(train.annotations)} anotações")
    print(f"  validação: {n_val} imagens, {len(val.annotations)} anotações")
    if test:
        print(f"  teste:     {n_test} imagens, {len(test.annotations)} anotações")
    cat_count = Counter()
    for a in train.annotations + val.annotations + (test.annotations if test else []):
        cat_count[a["category_id"]] += 1
    print("-" * 60)
    print("  classes:")
    for c in categories:
        print(f"    [{c['id']:>3}] {c['name']:<25} {cat_count.get(c['id'], 0):>6} anotações")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
