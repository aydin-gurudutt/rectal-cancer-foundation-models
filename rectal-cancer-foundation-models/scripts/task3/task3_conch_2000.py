#!/usr/bin/env python3
"""Patient-separated five-model tissue pseudo-label linear-probe comparison.

Read-only inputs: original CONCH manifest/index references, precomputed H5 embeddings,
patient split CSV. Test patients are never loaded.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

CLASSES = ('TUM', 'MUS', 'STR')
MODEL_NAMES = ('conch_v15', 'virchow2', 'phikon_v2', 'hoptimus1', 'musk')


def read_sample(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'split', 'case_id', 'slide_id', 'patch_index', 'sra_class'}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError('Sampled patches CSV missing required columns')
    rows = [r for r in rows if r['split'] in ('train', 'val') and r['sra_class'] in CLASSES]
    if not rows:
        raise RuntimeError('No train/val samples')
    return rows


def read_split(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {r['slide_id']: (r['case_id'], r['split']) for r in rows}
    if len(lookup) != len(rows):
        raise RuntimeError('Duplicate slide in split file')
    return lookup


def check_metadata(rows, splits):
    members = defaultdict(set)
    seen = set()
    for r in rows:
        slide = r['slide_id']
        if slide not in splits:
            raise RuntimeError(f'Sampled slide absent from verified splits: {slide}')
        case, split = splits[slide]
        if (case, split) != (r['case_id'], r['split']):
            raise RuntimeError(f'Patient/split mismatch: {slide}')
        members[split].add(case)
        key = (slide, r['patch_index'])
        if key in seen:
            raise RuntimeError(f'Duplicate sample patch: {key}')
        seen.add(key)
    if members['train'] & members['val']:
        raise RuntimeError('Train/val patient overlap')
    return {k: len(v) for k, v in members.items()}


def find_directories(feature_root):
    found = {}
    for model in MODEL_NAMES:
        path = feature_root / ('features_' + model)
        if not path.is_dir():
            raise RuntimeError(f'Expected directory missing: {path}\n'
                               f'Available folders: {[p.name for p in feature_root.iterdir() if p.is_dir()]}')
        found[model] = path
    return found


def attach_coordinates(rows, conch_dir):
    indices_by_slide = defaultdict(list)
    for i, r in enumerate(rows):
        indices_by_slide[r['slide_id']].append((i, int(r['patch_index'])))
    coordinates = [None] * len(rows)
    for slide, entries in sorted(indices_by_slide.items()):
        path = conch_dir / (slide + '.h5')
        if not path.exists():
            raise RuntimeError(f'CONCH file missing: {path}')
        with h5py.File(path, 'r') as f:
            coord_ds = f['coords']
            indices = sorted(set(idx for _, idx in entries))
            if min(indices) < 0 or max(indices) >= len(coord_ds):
                raise RuntimeError(f'Invalid CONCH row index: {slide}')
            coords = np.asarray(coord_ds[indices])
            lookup = {idx: (int(pair[0]), int(pair[1])) for idx, pair in zip(indices, coords)}
            for i, idx in entries:
                coordinates[i] = lookup[idx]
    if any(c is None for c in coordinates):
        raise RuntimeError('Internal coordinate lookup failure')
    return coordinates


def map_models(rows, coord_targets, feature_dirs):
    positions_by_slide = defaultdict(list)
    for pos, row in enumerate(rows):
        positions_by_slide[row['slide_id']].append(pos)
    mappings = {m: np.full(len(rows), -1, dtype=np.int64) for m in MODEL_NAMES}
    stats = {m: {'missing_slides': [], 'duplicate_coordinates': [], 'shape_errors': []}
             for m in MODEL_NAMES}
    for model, folder in feature_dirs.items():
        print(f'Checking coordinate alignment: {model}', flush=True)
        for slide, positions in sorted(positions_by_slide.items()):
            path = folder / (slide + '.h5')
            if not path.exists():
                stats[model]['missing_slides'].append(slide)
                continue
            with h5py.File(path, 'r') as f:
                if 'coords' not in f or 'features' not in f:
                    stats[model]['shape_errors'].append(slide)
                    continue
                coords = f['coords'][:]
                if len(coords) != len(f['features']) or coords.ndim != 2 or coords.shape[1] != 2:
                    stats[model]['shape_errors'].append(slide)
                    continue
                target = {coord_targets[pos] for pos in positions}
                coord_to_idx = {}
                duplicates = set()
                for idx, pair in enumerate(coords):
                    xy = (int(pair[0]), int(pair[1]))
                    if xy not in target:
                        continue
                    if xy in coord_to_idx:
                        duplicates.add(xy)
                    else:
                        coord_to_idx[xy] = idx
                if duplicates:
                    stats[model]['duplicate_coordinates'].append(slide)
                for pos in positions:
                    xy = coord_targets[pos]
                    if xy in coord_to_idx and xy not in duplicates:
                        mappings[model][pos] = coord_to_idx[xy]
    return mappings, stats


def prepare_audit(rows, coordinates, mappings, stats, output_dir):
    common = np.logical_and.reduce([mappings[m] >= 0 for m in MODEL_NAMES])
    count_by_partition = Counter((r['split'], r['sra_class']) for r in rows)
    common_by_partition = Counter((r['split'], r['sra_class']) for i, r in enumerate(rows) if common[i])
    match_by_model = {m: int(np.sum(mapping >= 0)) for m, mapping in mappings.items()}
    audit = {
        'models': list(MODEL_NAMES),
        'samples_total': len(rows),
        'matched_samples_per_model': match_by_model,
        'common_samples': int(np.sum(common)),
        'train_validation_counts': {
            f'{split}_{cls}': {
                'original': count_by_partition[(split, cls)],
                'common': common_by_partition[(split, cls)]
            } for split in ('train', 'val') for cls in CLASSES
        },
        'model_issues': stats,
        'selection': 'intersection of exactly equal slide IDs and (x,y) top-left coordinates',
        'test_accessed': False
    }
    outpath = output_dir / 'five_model_alignment_audit.json'
    with outpath.open('w') as f:
        json.dump(audit, f, indent=2)
    print('\n=== ALIGNMENT AUDIT ===')
    print('Total train/val patches:', len(rows))
    for m, n in match_by_model.items():
        print(f'{m}: {n}/{len(rows)} match')
    print('Common patches:', int(np.sum(common)))
    for key, item in audit['train_validation_counts'].items():
        print(key, item)
    for model, errors in stats.items():
        for key, values in errors.items():
            if values:
                print(f'{model}: {key} ({len(values)}): {values[:10]}')
    print('Saved audit:', outpath)
    return common, audit


def load_features(model, rows, common_positions, mapping, feature_dir):
    by_slide = defaultdict(list)
    for outpos, original_pos in enumerate(common_positions):
        slide = rows[int(original_pos)]['slide_id']
        by_slide[slide].append((outpos, int(mapping[int(original_pos)])))
    X = None
    for slide, entries in sorted(by_slide.items()):
        with h5py.File(feature_dir / (slide + '.h5'), 'r') as f:
            ds = f['features']
            indices = sorted({idx for _, idx in entries})
            vectors = np.asarray(ds[indices], dtype=np.float32)
            if X is None:
                X = np.empty((len(common_positions), vectors.shape[1]), dtype=np.float32)
            if vectors.shape[1] != X.shape[1]:
                raise RuntimeError(f'Inconsistent feature dimension for {model}: {slide}')
            idxlookup = {idx: j for j, idx in enumerate(indices)}
            for outpos, idx in entries:
                X[outpos] = vectors[idxlookup[idx]]
    if X is None or not np.isfinite(X).all():
        raise RuntimeError(f'Invalid features for model {model}')
    return X


def train_compare(rows, coordinates, common, audit, mappings, feature_dirs, output_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)

    # Require enough of the original sampled patches in every model to avoid severe attrition.
    for k, item in audit['train_validation_counts'].items():
        if item['common'] < 0.8 * item['original'] or item['common'] < 100:
            raise RuntimeError(f'Insufficient common coverage ({k}): {item}; audit first')
    positions = np.flatnonzero(common)
    selected_rows = [rows[int(pos)] for pos in positions]
    y = np.array([CLASSES.index(r['sra_class']) for r in selected_rows])
    train = np.array([r['split'] == 'train' for r in selected_rows], dtype=bool)
    val = np.array([r['split'] == 'val' for r in selected_rows], dtype=bool)
    assert np.any(train) and np.any(val) and np.all(train | val)
    train_patients = {r['case_id'] for r in selected_rows if r['split'] == 'train'}
    val_patients = {r['case_id'] for r in selected_rows if r['split'] == 'val'}
    assert not train_patients.intersection(val_patients)
    predictions = {}
    summary = []
    for model in ('conch_v15',):
        print(f'\n=== TRAINING {model} ===', flush=True)
        X = load_features(model, rows, positions, mappings[model], feature_dirs[model])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42))
        pipe.fit(X[train], y[train])
        pred = pipe.predict(X[val])
        report = classification_report(y[val], pred, labels=[0, 1, 2],
                                       target_names=CLASSES, output_dict=True, zero_division=0)
        metrics = {
            'model': model,
            'train_samples': int(train.sum()),
            'val_samples': int(val.sum()),
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'feature_dimensions': int(X.shape[1]),
            'balanced_accuracy': float(balanced_accuracy_score(y[val], pred)),
            'macro_f1': float(report['macro avg']['f1-score']),
            'per_class': {cls: report[cls] for cls in CLASSES},
            'confusion_matrix': confusion_matrix(y[val], pred, labels=[0, 1, 2]).tolist(),
            'label_source': 'SRA_provisional',
            'test_evaluated': False
        }
        summary.append(metrics)
        predictions[model] = pred
        joblib.dump(pipe, output_dir / f'task3_{model}_common_linear_probe.joblib')
        print(f'Balanced accuracy: {metrics["balanced_accuracy"]:.4f}; macro F1: {metrics["macro_f1"]:.4f}', flush=True)
        del X, pipe
    with (output_dir / 'five_model_common_validation_metrics.json').open('w') as f:
        json.dump({'models': summary, 'class_order': CLASSES,
                   'sampling': 'common coordinate intersection of original CONCH sampled patches',
                   'test_evaluated': False}, f, indent=2)
    with (output_dir / 'five_model_common_validation_summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'train_samples', 'val_samples', 'balanced_accuracy', 'macro_f1',
                         'TUM_f1', 'MUS_f1', 'STR_f1'])
        for s in summary:
            writer.writerow([s['model'], s['train_samples'], s['val_samples'],
                             s['balanced_accuracy'], s['macro_f1']]
                            + [s['per_class'][cls]['f1-score'] for cls in CLASSES])
    # CONCH-only rerun: do not regenerate the five-model review CSV.
    return

    # Prioritize MUS/STR disagreements for independent pathology review.
    val_positions = np.flatnonzero(val)
    candidates = []
    for val_i, sample_i in enumerate(val_positions):
        row = selected_rows[int(sample_i)]
        if row['sra_class'] not in ('MUS', 'STR'):
            continue
        preds = {model: CLASSES[int(predictions[model][val_i])] for model in MODEL_NAMES}
        errors = sum(label != row['sra_class'] for label in preds.values())
        if errors == 0:
            continue
        xy = coordinates[int(positions[int(sample_i)])]
        candidates.append((errors, row, xy, preds))
    candidates.sort(key=lambda t: (-t[0], t[1]['case_id'], t[1]['slide_id'], int(t[1]['patch_index'])))
    with (output_dir / 'muscle_stroma_review_candidates.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'slide_id', 'conch_patch_index', 'x', 'y', 'sra_class',
                         'number_models_disagreeing'] + [m + '_prediction' for m in MODEL_NAMES])
        for errors, row, xy, preds in candidates[:500]:
            writer.writerow([row['case_id'], row['slide_id'], row['patch_index'],
                             xy[0], xy[1], row['sra_class'], errors] + [preds[m] for m in MODEL_NAMES])
    print('\nSaved comparison and muscle/stroma review candidates.', flush=True)
    print('These are model/SRA disagreements, not verified labeling mistakes.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('audit', 'train'), required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/task3_tissue'))
    parser.add_argument('--features', type=Path, default=Path('outputs/paper1_coords_512/20.0x_512px_0px_overlap'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = read_sample(args.root / 'conch_linear_probe_sampled_patches.csv')
    splits = read_split(args.root / 'task3_slide_splits.csv')
    print('Train/val patient counts:', check_metadata(rows, splits), flush=True)
    dirs = find_directories(args.features)
    coordinates = attach_coordinates(rows, dirs['conch_v15'])
    mappings, stats = map_models(rows, coordinates, dirs)
    common, audit = prepare_audit(rows, coordinates, mappings, stats, args.root)
    if args.stage == 'train':
        train_compare(rows, coordinates, common, audit, mappings, dirs, args.root / 'conch_2000')


if __name__ == '__main__':
    main()
