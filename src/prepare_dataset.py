import argparse
import random
import time
from pathlib import Path


def load_classes(classes_path: Path):
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def make_label_file(label_path: Path, class_id: int):
    # If the image contains a single object (whole image), use full-image box.
    # Format: class x_center y_center width height (normalized)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def collect_image_paths(images_dir: Path):
    # Accept common image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Ignore pre-existing train/val splits to avoid double-processing
    ignore_folders = {"train", "val"}

    paths = []
    for p in images_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(images_dir)
        if rel.parts and rel.parts[0].lower() in ignore_folders:
            continue
        paths.append(p)
    return paths


def build_class_map(classes, normalize=True):
    # Map normalized class name -> class index
    if normalize:
        return {c.strip().lower(): i for i, c in enumerate(classes)}
    return {c: i for i, c in enumerate(classes)}


def normalize_name(s: str):
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def main():
    p = argparse.ArgumentParser(description="Prepare YOLOv8/YOLOv11 dataset from a nested-folder image structure")
    p.add_argument("--dataset", default="data/dataset(tubes)", help="Root dataset folder")
    p.add_argument("--train-pct", type=float, default=0.9, help="Fraction of images to use for training")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--dry-run", action="store_true", help="Print actions but don\'t copy/write files")
    p.add_argument("--default-class", default=None,
                   help="Class name or id to use for images that don't match a folder; if omitted, skipped")
    p.add_argument("--force", action="store_true", help="Regenerate labels even if they already exist")
    p.add_argument("--balance", action="store_true", help="Oversample minority classes in the train split (helps with imbalance)")
    p.add_argument("--balance-max-ratio", type=int, default=3, help="Max ratio between largest and smallest class after balancing")
    args = p.parse_args()

    base = Path(args.dataset).resolve()
    images_dir = base / "images"
    labels_dir = base / "labels"
    classes_file = base / "classes.txt"
    data_yaml = base / "data.yaml"

    classes = load_classes(classes_file)
    class_map = build_class_map(classes)

    # Allow a default fallback class by name or index
    default_class_id = None
    if args.default_class is not None:
        if args.default_class.isdigit():
            default_class_id = int(args.default_class)
        else:
            lookup = build_class_map(classes, normalize=False)
            default_class_id = lookup.get(args.default_class)
        if default_class_id is None or default_class_id < 0 or default_class_id >= len(classes):
            raise SystemExit(f"❌ Invalid default class: {args.default_class}. Valid names: {classes}")

    print(f"✅ Loaded {len(classes)} classes from {classes_file}")
    if default_class_id is not None:
        print(f"⚠️ Using default class id {default_class_id} ('{classes[default_class_id]}') for unmatched images")

    imgs = collect_image_paths(images_dir)
    if not imgs:
        raise SystemExit(f"❌ No images found under {images_dir}")

    # Ensure label folder exists
    labels_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for img in imgs:
        rel = img.relative_to(images_dir)
        # Determine class from first directory component, if present
        if len(rel.parents) > 1:
            folder = rel.parts[0]
        else:
            folder = None

        class_id = None
        if folder:
            normalized = normalize_name(folder)
            # Try match by normalized string
            for name, idx in class_map.items():
                if normalize_name(name) == normalized:
                    class_id = idx
                    break

        # Fallback: use default class if specified, else skip unclassified images
        if class_id is None:
            if default_class_id is not None:
                class_id = default_class_id
            else:
                print(f"⚠️ Skipping {img.name}: unable to resolve class (folder='{folder}')")
                continue

        label_path = labels_dir / (img.stem + ".txt")
        if args.force or not label_path.exists():
            if args.dry_run:
                print(f"[dry] would create label for {img.name} -> class {class_id}")
            else:
                make_label_file(label_path, class_id)
                print(f"Created label for {img.name}: class {class_id}")

        prepared.append((img, label_path))

    # Split into train/val
    random.seed(args.seed)
    random.shuffle(prepared)
    split_idx = int(len(prepared) * args.train_pct)
    train = prepared[:split_idx]
    val = prepared[split_idx:]

    def copy_split(items, split_name):
        dst_img = images_dir / split_name
        dst_lbl = labels_dir / split_name
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in items:
            dst_img_path = dst_img / img_path.name
            dst_lbl_path = dst_lbl / lbl_path.name
            if args.dry_run:
                print(f"[dry] would copy {img_path} -> {dst_img_path}")
                print(f"[dry] would copy {lbl_path} -> {dst_lbl_path}")
            else:
                from shutil import copy2
                copy2(img_path, dst_img_path)
                copy2(lbl_path, dst_lbl_path)

    print(f"Splitting {len(prepared)} images: {len(train)} train / {len(val)} val")
    copy_split(train, "train")
    copy_split(val, "val")

    # Optionally balance classes by oversampling the train split
    if args.balance:
        from shutil import copy2
        print("🧮 Balancing classes via oversampling in train split...")
        train_lbl_dir = labels_dir / "train"
        train_img_dir = images_dir / "train"

        # Group by class
        class_to_labels = {}
        for label_file in train_lbl_dir.glob("*.txt"):
            try:
                cls = int(label_file.read_text().strip().split()[0])
            except Exception:
                continue
            class_to_labels.setdefault(cls, []).append(label_file)

        if not class_to_labels:
            print("⚠️ No labels found in train split; skipping balancing")
        else:
            counts = {cls: len(files) for cls, files in class_to_labels.items()}
            min_count = min(counts.values())
            max_count = max(counts.values())
            target = min(max_count, min_count * args.balance_max_ratio)
            print(f"Class counts before balancing: {counts}")
            print(f"Target per-class count (max ratio {args.balance_max_ratio}): {target}")

            # Oversample to target
            for cls, label_files in class_to_labels.items():
                current = len(label_files)
                if current >= target:
                    continue
                needed = target - current
                for i in range(needed):
                    src_label = random.choice(label_files)
                    stem = src_label.stem
                    # Find matching image by stem (any extension)
                    candidates = list(train_img_dir.glob(f"{stem}.*"))
                    if not candidates:
                        continue
                    src_img = candidates[0]
                    dup_suffix = f"_dup{int(time.time()*1000)%100000}_{i}"
                    dst_img = train_img_dir / f"{stem}{dup_suffix}{src_img.suffix}"
                    dst_lbl = train_lbl_dir / f"{stem}{dup_suffix}.txt"
                    if not args.dry_run:
                        copy2(src_img, dst_img)
                        copy2(src_label, dst_lbl)
                        label_files.append(dst_lbl)
            print("✅ Balancing complete.")

    # Update data.yaml to point to new splits
    yaml_text = f"path: {base.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for i, name in enumerate(classes):
        yaml_text += f"  {i}: '{name}'\n"

    if args.dry_run:
        print("[dry] would update data.yaml with:\n" + yaml_text)
    else:
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        print(f"Updated {data_yaml}")

    print("✅ Dataset prepared (train/val split + labels).")


if __name__ == "__main__":
    main()
