import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np


def split_data_mixed(
    input_dir: str,
    output_dir: str = "data",
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
    clean_output: bool = True,
    class_names: Tuple[str, str] = ("drowsy", "notdrowsy"),
) -> Tuple[List[str], List[str], List[str]]:
    """
    Person- and class-aware split (e.g., drowsy/notdrowsy) with 60/20/20 ratios.

    For each PERSON and for each CLASS independently:
      1) Shuffle that person's files of the class.
      2) Split into train/val/test by the given ratios (rounded).
      3) Copy to output_dir/{train,val,test}/ and append a numeric suffix to the filename:
         '_1' for 'drowsy' and '_0' for 'notdrowsy'.

    Expected input directory structure:
        input_dir/
            drowsy/*.jpg      (e.g., 001_xxx.jpg  -> person id = '001')
            notdrowsy/*.jpg

    Returns:
        (train_file_paths, val_file_paths, test_file_paths) within the output_dir.
    """
    # --- basic validations ---
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-8, "train_ratio + val_ratio + test_ratio must sum to 1.0"

    rng = random.Random(seed)

    # --- prepare output directories ---
    train_dir = os.path.join(output_dir, "train")
    val_dir   = os.path.join(output_dir, "val")
    test_dir  = os.path.join(output_dir, "test")

    if clean_output and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    for d in (train_dir, val_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    # --- scan input: person_id -> class_name -> list[file_path] ---
    # person_id is taken as the substring before the first '_' in the filename (e.g., '001_something.jpg' -> '001').
    people_to_class_files: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for cls in class_names:
        class_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.startswith("."):
                continue
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Extract person id from filename
            parts = fname.split("_", 1)
            person_id = parts[0] if parts else fname  # fallback: whole name if no underscore
            people_to_class_files[person_id][cls].append(fpath)

    if not people_to_class_files:
        raise ValueError(
            "No data found. Check your folder structure: input_dir/{drowsy,notdrowsy} should contain files."
        )

    # --- helper: split a list of items into (train, val, test) using rounded counts ---
    def split_by_ratios_round(items: List[str], tr: float, va: float, te: float) -> Tuple[List[str], List[str], List[str]]:
        n = len(items)
        if n == 0:
            return [], [], []
        n_train = int(np.round(n * tr))
        n_val   = int(np.round(n * va))
        n_test  = n - n_train - n_val  # ensure sum equals n
        # In rare cases rounding can make n_test negative for tiny n; clamp it and adjust val/train if needed
        if n_test < 0:
            # reduce val first, then train
            deficit = -n_test
            take_from_val = min(deficit, n_val)
            n_val -= take_from_val
            deficit -= take_from_val
            if deficit > 0:
                n_train = max(0, n_train - deficit)
            n_test = n - n_train - n_val
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    # --- helper: copy and append numeric class id suffix (_1 for drowsy, _0 for notdrowsy) ---
    class_to_id = {"drowsy": 1, "notdrowsy": 0}

    def append_class_id_suffix_and_copy(src_paths: List[str], dst_dir: str):
        for src in src_paths:
            cls = Path(src).parent.name
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            # if the source already ends with _0 or _1, strip it to avoid double-suffixing
            if name.endswith(("_0", "_1")) and len(name) > 2:
                name = name[:-2]
            new_name = f"{name}_{class_to_id.get(cls, -1)}{ext}"
            dst = os.path.join(dst_dir, new_name)
            shutil.copy2(src, dst)

    # --- main loop: for each person, for each class (drowsy first, then notdrowsy) ---
    train_srcs, val_srcs, test_srcs = [], [], []

    for person_id in sorted(people_to_class_files.keys()):
        for cls in class_names:  # order matters: ('drowsy', 'notdrowsy')
            files = list(people_to_class_files[person_id][cls])
            rng.shuffle(files)
            tr_list, va_list, te_list = split_by_ratios_round(files, train_ratio, val_ratio, test_ratio)

            append_class_id_suffix_and_copy(tr_list, train_dir)
            append_class_id_suffix_and_copy(va_list, val_dir)
            append_class_id_suffix_and_copy(te_list, test_dir)

            train_srcs.extend(tr_list)
            val_srcs.extend(va_list)
            test_srcs.extend(te_list)

    # --- summary printout (sanity check) ---
    def summarize(paths: List[str]) -> Tuple[Dict[str, int], Dict[str, int], int]:
        person_counts = Counter([Path(p).name.split("_")[0] for p in paths])
        class_counts  = Counter([Path(p).parent.name for p in paths])
        return dict(person_counts), dict(class_counts), len(paths)

    tr_persons, tr_classes, tr_n = summarize(train_srcs)
    va_persons, va_classes, va_n = summarize(val_srcs)
    te_persons, te_classes, te_n = summarize(test_srcs)

    print(f"Train: {tr_n} | Persons: {tr_persons} | Classes: {tr_classes}")
    print(f"Val:   {va_n} | Persons: {va_persons} | Classes: {va_classes}")
    print(f"Test:  {te_n} | Persons: {te_persons} | Classes: {te_classes}")

    # return the resulting file paths inside output dirs
    return (
        [os.path.join(train_dir, f) for f in os.listdir(train_dir)],
        [os.path.join(val_dir, f) for f in os.listdir(val_dir)],
        [os.path.join(test_dir, f) for f in os.listdir(test_dir)],
    )
