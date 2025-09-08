# Person-wise split and append _0/_1 to filenames

import os
import shutil
import random
from typing import List
import numpy as np

def split_data_by_person(
    train_data_dir: str,
    output_dir: str = "data_by_person",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    clean_output: bool = False,
    labels: List[str] = ("drowsy", "notdrowsy"),
):
    """
    Split by person into train/val/test and copy to target folders.
    """
    # Prepare output folders
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    if clean_output and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Labels and numeric equivalents
    label_to_id = {'drowsy': 1, 'notdrowsy': 0}

    # Group by person
    people = {}
    for label in labels:
        label_path = os.path.join(train_data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            if filename.startswith('.'):
                continue
            full_path = os.path.join(label_path, filename)
            if not os.path.isfile(full_path):
                continue
            person_id = filename.split('_')[0]  # First token is person id
            if person_id not in people:
                people[person_id] = []
            people[person_id].append((full_path, label))

    # Shuffle persons
    person_ids = list(people.keys())
    random.Random(seed).shuffle(person_ids)

    # Split persons by ratios
    n = len(person_ids)
    
    if n < 3:
        print(f"⚠️  Warning: Only {n} people! Using minimum split...")
        if n == 1:
            train_persons = person_ids
            val_persons = []
            test_persons = []
        elif n == 2:
            train_persons = [person_ids[0]]
            val_persons = []
            test_persons = [person_ids[1]]
    else:
        # Normal split
        train_size = int(np.round(train_ratio * n))
        val_size = int(np.round(val_ratio * n))
        test_size = int(np.round(test_ratio * n))
        train_persons = person_ids[:train_size]
        val_persons = person_ids[train_size:train_size + val_size]
        test_persons = person_ids[train_size + val_size:train_size + val_size + test_size]

    # Copy data to target folders
    def copy_for(persons, target_dir):
        paths = []
        for pid in persons:
            for src_path, label in people[pid]:
                base = os.path.basename(src_path)
                name, ext = os.path.splitext(base)
                numeric = label_to_id[label]
                new_name = f"{name}_{numeric}{ext}"
                dst_path = os.path.join(target_dir, new_name)
                shutil.copy2(src_path, dst_path)
                paths.append(dst_path)
        return paths

    train_paths = copy_for(train_persons, train_dir)
    val_paths = copy_for(val_persons, val_dir)
    test_paths = copy_for(test_persons, test_dir)

    print(f"Train: {len(train_persons)} person, {len(train_paths)} image")
    print(f"Validation: {len(val_persons)} person, {len(val_paths)} image")
    print(f"Test: {len(test_persons)} person, {len(test_paths)} image")

    return train_paths, val_paths, test_paths