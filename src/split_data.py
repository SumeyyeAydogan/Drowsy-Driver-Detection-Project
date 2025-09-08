import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import numpy as np

def split_data(
    train_data_dir: str,
    output_dir: str = "data",
    val_ratio: float = 0.30,   # 30% for validation from remaining
    test_ratio: float = 0.20,  # 20% for test from persons
    seed: int = 42,
    clean_output: bool = False,
    labels: Tuple[str, str] = ("drowsy", "notdrowsy"),
    person_cap: float = 0.50,  # not used but keeping for signature
):
    """
    1) First, 20% of the persons are assigned to the TEST set.
    2) 30% of the remaining persons are assigned to the VALIDATION set (person-specific drowsy/notdrowsy ratio is maintained).
    3) The rest goes to the TRAIN set.

    Data structure:
        train_data_dir/
            drowsy/*.jpg (e.g. 001_....jpg)
            notdrowsy/*.jpg

    Output:
        Files are copied into output_dir/{train,val,test}
    """

    # --- Helpers ---
    def _ensure_dirs():
        if clean_output and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        train_dir = os.path.join(output_dir, 'train')
        val_dir   = os.path.join(output_dir, 'val')
        test_dir  = os.path.join(output_dir, 'test')
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)
        return train_dir, val_dir, test_dir

    def _scan_people(root: str, labels: Tuple[str, str]):
        people: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for label in labels:
            label_path = os.path.join(root, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                if fname.startswith('.'):
                    continue
                fpath = os.path.join(label_path, fname)
                if not os.path.isfile(fpath):
                    continue
                person_id = fname.split('_')[0]
                people[person_id][label].append(fpath)
        return people

    def _rand_round(x: float, rng: random.Random) -> int:
        base = int(np.floor(x))
        frac = x - base
        return base + (1 if rng.random() < frac else 0)

    def _initial_val_selection(people, val_ratio: float, seed: int):
        rng = random.Random(seed)
        val_pick = defaultdict(lambda: defaultdict(list))
        for pid, per_label in people.items():
            for label, items in per_label.items():
                items = list(items)
                rng.shuffle(items)
                k = _rand_round(val_ratio * len(items), rng)
                k = max(0, min(k, len(items)))
                val_pick[pid][label] = items[:k]
        return val_pick

    def _copy(samples: List[str], target_dir: str, label_to_id: Dict[str, int]):
        for src_path in samples:
            label = Path(src_path).parent.name
            base = os.path.basename(src_path)
            name, ext = os.path.splitext(base)
            numeric = label_to_id.get(label, -1)
            new_name = f"{name}_{numeric}{ext}"
            dst_path = os.path.join(target_dir, new_name)
            shutil.copy2(src_path, dst_path)

    def _summarize(samples: List[str]):
        per_person = Counter([Path(x).name.split('_')[0] for x in samples])
        per_label  = Counter([Path(x).parent.name for x in samples])
        return per_person, per_label, len(samples)

    def _summarize_per_person(samples: List[str]):
        summary = defaultdict(lambda: Counter())
        for x in samples:
            pid = Path(x).name.split('_')[0]
            label = Path(x).parent.name
            summary[pid][label] += 1
        return summary

    # --- Main process ---
    train_dir, val_dir, test_dir = _ensure_dirs()
    label_to_id = {'drowsy': 1, 'notdrowsy': 0}

    people = _scan_people(train_data_dir, labels)
    if not people:
        raise ValueError("No data found. Please check the folders and label names.")

    rng = random.Random(seed)
    person_ids = sorted(people.keys())
    rng.shuffle(person_ids)

    n_total = len(person_ids)
    n_test = _rand_round(test_ratio * n_total, rng)
    test_persons = set(person_ids[:n_test])
    remain_persons = {pid: people[pid] for pid in person_ids[n_test:]}

    # Validation selection (maintains person-specific drowsy/notdrowsy ratio)
    val_pick = _initial_val_selection(remain_persons, val_ratio, seed)

    # Split samples into train, val, and test
    test_samples, val_samples, train_samples = [], [], []
    for pid, per_label in people.items():
        for label, items in per_label.items():
            if pid in test_persons:
                test_samples.extend(items)
            elif pid in val_pick:
                chosen = val_pick[pid][label]
                val_samples.extend(chosen)
                train_samples.extend([x for x in items if x not in chosen])
            else:
                train_samples.extend(items)

    # Copy files to corresponding directories
    _copy(train_samples, train_dir, label_to_id)
    _copy(val_samples,   val_dir,   label_to_id)
    _copy(test_samples,  test_dir,  label_to_id)

    # --- Console outputs ---
    # Starting summary
    all_samples = [x for pid in people.values() for label in pid.values() for x in label]
    all_persons, all_labels, all_n = _summarize(all_samples)
    print("\n--- START ---")
    print(f"Total persons: {len(all_persons)} | Total images: {all_n}")
    print(f"Person-wise samples: {dict(all_persons)}")
    print(f"Class distribution: {dict(all_labels)}")

    # Results
    tr_pers, tr_lab, tr_n = _summarize(train_samples)
    va_pers, va_lab, va_n = _summarize(val_samples)
    te_pers, te_lab, te_n = _summarize(test_samples)

    print("\n--- PLANNED splits ---")
    print(f"Test: {test_ratio:.0%} (person-based)")
    print(f"Val:  {val_ratio:.0%} (from remaining, maintaining person-specific ratio)")
    print("Train: the rest")

    print("\n--- ACTUAL splits (Person-based) ---")
    print(f"Train persons: {len(tr_pers)} / {n_total} ({len(tr_pers)/n_total:.2%})")
    print(f"Val persons:   {len(va_pers)} / {n_total} ({len(va_pers)/n_total:.2%})")
    print(f"Test persons:  {len(te_pers)} / {n_total} ({len(te_pers)/n_total:.2%})")

    print("\n--- ACTUAL splits (Image & class-wise) ---")
    total_samples = tr_n + va_n + te_n
    print(f"Train: {tr_n} images ({tr_n/total_samples:.2%}) | classes: {dict(tr_lab)}")
    print(f"Val:   {va_n} images ({va_n/total_samples:.2%}) | classes: {dict(va_lab)}")
    print(f"Test:  {te_n} images ({te_n/total_samples:.2%}) | classes: {dict(te_lab)}")

    # --- Detailed person-based distribution ---
    print("\n--- DETAILED DISTRIBUTION ---")
    for set_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        print(f"\n{set_name}/")
        per_person = _summarize_per_person(samples)
        for pid, counts in per_person.items():
            counts_str = " ".join([f"{label}:{cnt}" for label, cnt in counts.items()])
            print(f"  Person {pid} -> {counts_str}")

    return (
        [os.path.join(train_dir, f) for f in os.listdir(train_dir)],
        [os.path.join(val_dir, f) for f in os.listdir(val_dir)],
        [os.path.join(test_dir, f) for f in os.listdir(test_dir)],
    )
