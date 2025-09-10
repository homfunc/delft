import h5py
import numpy as np

def load_weights_by_name_from_h5(model, filepath, verbose=True, strict=True, allow_multiple=False):
    """
    Load weights from a legacy Keras HDF5 file by matching variable names.

    Args:
        model: Keras model whose .weights are to be assigned.
        filepath: Path to legacy HDF5 file.
        verbose: If True, prints a short mapping/missing summary.
        strict: If True, raises ValueError when some model weights are not matched.
        allow_multiple: If False, raises ValueError when multiple legacy datasets match a single model weight.

    Returns:
        (assigned_count, missing_list) where missing_list is a list of (var_name, shape).
    """
    with h5py.File(filepath, 'r') as f:
        # Build map of dataset name -> np array
        datasets = {}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = np.array(obj)
        f.visititems(lambda name, obj: visit(name, obj))
    # Also build a list of keys without trailing ':0'
    dataset_keys = list(datasets.keys())
    dataset_keys_nocolon = {k: (k[:-2] if k.endswith(':0') else k) for k in dataset_keys}
    # Build reverse lookup without ':0' to support exact-path matches from candidates()
    nocolon_to_orig = {dataset_keys_nocolon[k]: k for k in dataset_keys}
    datasets_nocolon = {dataset_keys_nocolon[k]: datasets[k] for k in dataset_keys}

    def candidates(var_name):
        import re
        # Strip trailing :0
        base = var_name[:-2] if var_name.endswith(":0") else var_name
        yield base
        # Drop first scope element
        parts = base.split('/')
        if len(parts) > 1:
            yield '/'.join(parts[1:])
        # Swap common outer scopes
        swaps = [
            ("functional/", "model/"),
            ("model/", "functional/"),
            ("crf_wrapper/", ""),
            ("", "crf_wrapper/"),
        ]
        for a, b in swaps:
            if a and base.startswith(a):
                yield b + base[len(a):]
        # Normalize CRF scopes: allow 'crf_wrapper/crf' <-> 'crf/crf' <-> 'crf'
        norm_crf = [
            base,
            base.replace('crf_wrapper/crf/', 'crf/crf/'),
            base.replace('crf_wrapper/crf/', 'crf/'),
            base.replace('crf/', 'crf/crf/'),
        ]
        for v in norm_crf:
            yield v
        # For inner CRF projections, allow dense vs dense_1 swaps
        if '/dense_1/' in base:
            yield base.replace('/dense_1/', '/dense/')
        if '/dense/' in base:
            yield base.replace('/dense/', '/dense_1/')
        # CRF naming synonyms across implementations
        # transitions can be named 'transitions', 'chain_kernel', or 'U'
        if base.endswith('/transitions'):
            yield base.replace('/transitions', '/chain_kernel')
            yield base.replace('/transitions', '/U')
        if base.endswith('/chain_kernel'):
            yield base.replace('/chain_kernel', '/transitions')
            yield base.replace('/chain_kernel', '/U')
        if base.endswith('/U'):
            yield base.replace('/U', '/transitions')
            yield base.replace('/U', '/chain_kernel')
        # Boundary parameters sometimes named left_boundary/right_boundary vs b_start/b_end
        if base.endswith('/left_boundary'):
            yield base.replace('/left_boundary', '/b_start')
        if base.endswith('/right_boundary'):
            yield base.replace('/right_boundary', '/b_end')
        if base.endswith('/b_start'):
            yield base.replace('/b_start', '/left_boundary')
        if base.endswith('/b_end'):
            yield base.replace('/b_end', '/right_boundary')
        # Char embedding mapping: time_distributed/char_embeddings/embeddings -> model/time_distributed/embeddings
        if base.endswith('time_distributed/char_embeddings/embeddings'):
            yield base.replace('time_distributed/char_embeddings/embeddings', 'model/time_distributed/embeddings')
        if '/time_distributed/char_embeddings/embeddings' in base:
            yield base.replace('/time_distributed/char_embeddings/embeddings', '/time_distributed/embeddings')
        # TimeDistributed BiLSTM mapping inside bidirectional: forward/backward cell indices
        m = re.match(r"(.*/time_distributed_1)/bidirectional/(forward_lstm|backward_lstm)/lstm_cell(?:_\d+)?/(kernel|recurrent_kernel|bias)$", base)
        if m:
            root, dirn, param = m.group(1), m.group(2), m.group(3)
            cell_idx = 'lstm_cell_1' if dirn == 'forward_lstm' else 'lstm_cell_2'
            yield f"model/{root.split('/',1)[1]}/{dirn}/{cell_idx}/{param}"
            yield f"{root}/{dirn}/{cell_idx}/{param}"
        # Upgraded explicit char LSTMs -> legacy TimeDistributed BiLSTM mapping
        m_ch_f = re.match(r".*/char_lstm_fwd/(?:lstm(?:_\d+)?)/lstm_cell(?:_\d+)?/(kernel|recurrent_kernel|bias)$", base)
        if m_ch_f:
            param = m_ch_f.group(1)
            yield f"model/time_distributed_1/forward_lstm/lstm_cell_1/{param}"
            yield f"time_distributed_1/forward_lstm/lstm_cell_1/{param}"
        m_ch_b = re.match(r".*/char_lstm_bwd/(?:lstm(?:_\d+)?)/lstm_cell(?:_\d+)?/(kernel|recurrent_kernel|bias)$", base)
        if m_ch_b:
            param = m_ch_b.group(1)
            yield f"model/time_distributed_1/backward_lstm/lstm_cell_2/{param}"
            yield f"time_distributed_1/backward_lstm/lstm_cell_2/{param}"
        # Main BiLSTM mapping: forward/backward have numeric suffixes in legacy (4/5)
        m2 = re.match(r"(.*/bidirectional_1)/(forward_lstm(?:_\d+)?|backward_lstm(?:_\d+)?)/lstm_cell(?:_\d+)?/(kernel|recurrent_kernel|bias)$", base)
        if m2:
            root, dirn, param = m2.group(1), m2.group(2), m2.group(3)
            cell_idx = 'lstm_cell_4' if dirn.startswith('forward') else 'lstm_cell_5'
            yield f"model/{root.split('/',1)[1]}/{dirn}/{cell_idx}/{param}"
            yield f"{root}/{dirn}/{cell_idx}/{param}"
        if m_ch_b:
            param = m_ch_b.group(1)
            yield f"model/time_distributed_1/backward_lstm/lstm_cell_2/{param}"
            yield f"time_distributed_1/backward_lstm/lstm_cell_2/{param}"
        # Main BiLSTM mapping: forward/backward have numeric suffixes in legacy (4/5)
        m2 = re.match(r"(.*/bidirectional_1)/(forward_lstm(?:_\d+)?|backward_lstm(?:_\d+)?)/lstm_cell(?:_\d+)?/(kernel|recurrent_kernel|bias)$", base)
        if m2:
            root, dirn, param = m2.group(1), m2.group(2), m2.group(3)
            cell_idx = 'lstm_cell_4' if dirn.startswith('forward') else 'lstm_cell_5'
            yield f"model/{root.split('/',1)[1]}/{dirn}/{cell_idx}/{param}"
            yield f"{root}/{dirn}/{cell_idx}/{param}"

    assigned = 0
    missing = []
    mapping_log = []
    for var in model.weights:
        found = False
        shape = tuple(var.shape)
        var_name = getattr(var, 'name', '')
        var_path = getattr(var, 'path', var_name)
        var_base = var_path[:-2] if var_path.endswith(":0") else var_path
        for key in candidates(var_path):
            # Prefer exact path (segment-aware) matches, ignoring trailing ':0'
            if key in datasets_nocolon:
                arr = datasets_nocolon[key]
                if arr.shape == shape:
                    orig_key = nocolon_to_orig[key]
                    var.assign(datasets[orig_key])
                    mapping_log.append((var_path, orig_key))
                    found = True
                    assigned += 1
                    break
        if not found:
            # suffix match as last resort (with explicit legacy alias mapping)
            base = var_base
            # Build candidate suffixes including legacy aliases (no scope), and with current scope
            suffixes = set()
            parts = base.split('/')
            last = parts[-1]
            # original forms
            suffixes.add(base)
            suffixes.add(last)
            # legacy aliases for CRF variables
            alias_map = {
                'transitions': ['U', 'chain_kernel'],
                'left_boundary': ['b_start'],
                'right_boundary': ['b_end'],
            }
            for key, aliases in alias_map.items():
                if base.endswith('/' + key):
                    for a in aliases:
                        suffixes.add(a)
                        suffixes.add('/'.join(parts[:-1] + [a]))
                if last == key:
                    for a in aliases:
                        suffixes.add(a)
            import re

            def extract_dir_token(s: str):
                m = re.search(r'/(forward_lstm(?:_\d+)?|backward_lstm(?:_\d+)?)/', s)
                if not m:
                    # Heuristic for upgraded layer names
                    if 'char_lstm_fwd' in s:
                        return 'forward'
                    if 'char_lstm_bwd' in s:
                        return 'backward'
                    return None
                tok = m.group(1)
                return 'forward' if tok.startswith('forward') else 'backward'

            def suffix_segment_match(path: str, suffix: str) -> bool:
                # Segment-aware suffix matching: compare full segments, not substrings
                p_parts = path.split('/')
                s_parts = suffix.split('/')
                if len(s_parts) == 1:
                    return bool(p_parts) and p_parts[-1] == s_parts[0]
                if len(s_parts) <= len(p_parts):
                    return p_parts[-len(s_parts):] == s_parts
                return False

            def normalize_segments(path: str):
                # Remove numeric suffixes like _1, _2 from each segment for semantic comparison
                parts = path.split('/')
                norm = []
                for seg in parts:
                    if seg in ('model', 'functional'):
                        norm.append(seg)
                        continue
                    seg2 = re.sub(r'_(?:\d+)$', '', seg)
                    norm.append(seg2)
                return norm

            def trailing_match_score(path_a: str, path_b: str) -> int:
                a = normalize_segments(path_a)
                b = normalize_segments(path_b)
                score = 0
                for sa, sb in zip(reversed(a), reversed(b)):
                    if sa == sb:
                        score += 1
                    else:
                        break
                return score

            def token_overlap_score(path_a: str, path_b: str) -> int:
                a = set(normalize_segments(path_a))
                b = set(normalize_segments(path_b))
                return len(a.intersection(b))

            def length_diff_score(path_a: str, path_b: str) -> int:
                return abs(len(normalize_segments(path_a)) - len(normalize_segments(path_b)))

            # Group-level direction consistency: keep same direction for kernel/recurrent_kernel/bias
            def group_norm(path: str) -> str:
                parts = path.split('/')
                parent = '/'.join(parts[:-1]) if len(parts) > 1 else ''
                return '/'.join(normalize_segments(parent))

            matches = []
            var_dir = extract_dir_token(base)
            var_group = group_norm(base)
            assigned_groups = getattr(load_weights_by_name_from_h5, '_assigned_groups', {})
            for k, v in datasets.items():
                if v.shape != shape:
                    continue
                k_nc = dataset_keys_nocolon[k]
                ds_dir = extract_dir_token(k_nc)
                # Filter by direction when available on both sides
                if var_dir and ds_dir and var_dir != ds_dir:
                    continue
                seg_ok = False
                for suf in suffixes:
                    if suffix_segment_match(k_nc, suf):
                        seg_ok = True
                        break
                if not seg_ok:
                    continue
                matches.append(k)

            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                # Apply group-level direction consistency if previously chosen
                prev_dir = assigned_groups.get(var_group)
                if prev_dir:
                    dir_filtered = [k for k in matches if extract_dir_token(dataset_keys_nocolon[k]) == prev_dir]
                    if dir_filtered:
                        matches = dir_filtered
                # Prefer 'model/' prefix, then highest trailing-match score
                preferred = [k for k in matches if k.startswith('model/')]
                pool = preferred or matches
                # Score by trailing segment alignment after normalization (strip _n)
                scores = [(k, trailing_match_score(dataset_keys_nocolon[k], base)) for k in pool]
                max_score = max(s for _, s in scores) if scores else 0
                pool = [k for k, s in scores if s == max_score]
                if len(pool) > 1:
                    # Next tiebreaker: overall token overlap
                    ov_scores = [(k, token_overlap_score(dataset_keys_nocolon[k], base)) for k in pool]
                    max_ov = max(s for _, s in ov_scores)
                    pool = [k for k, s in ov_scores if s == max_ov]
                if len(pool) > 1:
                    # Next tiebreaker: choose path with minimal length difference to base
                    len_scores = [(k, length_diff_score(dataset_keys_nocolon[k], base)) for k in pool]
                    min_len = min(s for _, s in len_scores)
                    pool = [k for k, s in len_scores if s == min_len]
                if len(pool) > 1:
                    # Final neutral deterministic choice (no forward/backward bias): lexicographic
                    pool = sorted(pool)
                target = pool[0]
            else:
                target = None

            if target is not None:
                arr = datasets[target]
                var.assign(arr)
                mapping_log.append((var_path, target))
                # Remember chosen direction for this group if any
                ds_dir = extract_dir_token(dataset_keys_nocolon[target])
                if ds_dir:
                    assigned_groups[var_group] = ds_dir
                    setattr(load_weights_by_name_from_h5, '_assigned_groups', assigned_groups)
                found = True
                assigned += 1
        if not found:
            missing.append((var.name, shape))
    if verbose:
        print(f"Assigned {assigned} variables from legacy file; missing {len(missing)}")
        if mapping_log:
            print("Sample mappings:")
            for n, k in mapping_log[:]:
                print("  ", n, "<=", k)
        if missing:
            print("Missing examples:")
            for n, s in missing[:12]:
                print("  ", n, s)
    if strict and missing:
        missing_str = "\n".join([f"- {n} {s}" for n, s in missing])
        raise ValueError(f"Strict load failed: {len(missing)} model weights were not matched.\n{missing_str}")
    return assigned, missing

