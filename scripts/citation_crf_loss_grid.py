#!/usr/bin/env python3
import os
# Set TF threads low for memory stability before TF loads
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import time
import argparse
from typing import Optional, List, Tuple

from sklearn.model_selection import train_test_split

from delft.sequenceLabelling.reader import load_data_and_labels_crf_file
from delft.sequenceLabelling.wrapper import Sequence
from delft.sequenceLabelling.trainer import Scorer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def evaluate_micro_f1(model: Sequence, x_test, y_test, features=None) -> float:
    """
    Mirror the logic in wrapper.eval_single to compute micro F1 using Scorer, but return the float.
    """
    # Prepare generator
    generator = model.model.get_generator()
    test_generator = generator(
        x_test,
        y_test,
        batch_size=model.model_config.batch_size,
        preprocessor=model.p,
        char_embed_size=model.model_config.char_embedding_size,
        max_sequence_length=model.model_config.max_sequence_length,
        embeddings=model.embeddings,
        shuffle=False,
        features=features,
        output_input_offsets=True,
        use_chain_crf=model.model_config.use_chain_crf,
        pad_to_max_sequence_length=False,
    )
    scorer = Scorer(
        test_generator,
        model.p,
        evaluation=True,
        use_crf=model.model_config.use_crf,
        use_chain_crf=model.model_config.use_chain_crf,
        bound_model=model.model,
    )
    scorer.on_epoch_end(epoch=-1)
    return float(scorer.f1 if scorer.f1 is not None else 0.0)


def run_one(train_file: str,
            eval_file: str,
            loss_type: str,
            alpha: Optional[float],
            smooth: Optional[float],
            batch_size: int,
            max_seq_len: int,
            max_epoch: int,
            early_stop: bool,
            patience: int,
            embeddings_name: str,
            csv_path: str,
            seed: int = 42) -> float:
    # Load train data and split train/valid (90/10)
    x_all, y_all, f_all = load_data_and_labels_crf_file(train_file)
    x_train, x_valid, y_train, y_valid, f_train, f_valid = train_test_split(
        x_all, y_all, f_all, test_size=0.1, shuffle=True, random_state=seed
    )

    # Build model
    model = Sequence(
        model_name='grobid-citation-BidLSTM_CRF',
        architecture='BidLSTM_CRF',
        embeddings_name=embeddings_name,
        batch_size=batch_size,
        max_sequence_length=max_seq_len,
        learning_rate=0.001,
        max_epoch=max_epoch,
        early_stop=early_stop,
        patience=patience,
        transformer_name=None,
        crf_loss_type=loss_type,
        crf_dice_smooth=(smooth if smooth is not None else 1.0),
        crf_joint_nll_weight=(alpha if alpha is not None else 0.2),
        crf_use_boundary=True,
        report_to_wandb=False,
    )

    # Train
    t0 = time.time()
    model.train(x_train, y_train, f_train=f_train, x_valid=x_valid, y_valid=y_valid, f_valid=f_valid, incremental=False, callbacks=None)
    train_seconds = round(time.time() - t0)

    # Load eval data and evaluate
    x_eval, y_eval, f_eval = load_data_and_labels_crf_file(eval_file)
    f1_micro = evaluate_micro_f1(model, x_eval, y_eval, features=f_eval)

    # Append to CSV
    ensure_dir(os.path.dirname(csv_path))
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(['loss_type', 'alpha', 'smooth', 'epochs', 'batch_size', 'f1_micro', 'train_seconds'])
        writer.writerow([
            loss_type,
            '' if alpha is None else alpha,
            '' if smooth is None else smooth,
            max_epoch,
            batch_size,
            round(f1_micro * 100.0, 2),
            train_seconds,
        ])

    return f1_micro


def main():
    ap = argparse.ArgumentParser(description='Run CRF loss grid for GROBID citation (BidLSTM_CRF + glove).')
    ap.add_argument('--train-file', default='data/sequenceLabelling/grobid/citation/citation-060518.train')
    ap.add_argument('--eval-file', default='data/sequenceLabelling/grobid/citation/citation-231022.train')
    ap.add_argument('--batch-size', type=int, default=30)
    ap.add_argument('--max-seq-len', type=int, default=500)
    ap.add_argument('--max-epoch', type=int, default=20)
    ap.add_argument('--early-stop', type=str, default='true', help='true/false')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--embeddings', default='glove-840B')
    ap.add_argument('--csv-path', default='data/models/sequenceLabelling/grobid-citation-BidLSTM_CRF/crf_grid_results.csv')
    ap.add_argument('--alpha', nargs='*', type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument('--smooth', nargs='*', type=float, default=[0.1, 0.5, 1.0])
    args = ap.parse_args()

    early_stop = str(args.early_stop).lower() in ('1', 'true', 'yes', 'y')

    # Grid definition per user request
    jobs: List[Tuple[str, Optional[float], Optional[float]]] = []
    # NLL only
    jobs.append(('nll', None, None))
    # Dice only: sweep smooth
    for s in args.smooth:
        jobs.append(('dice', None, float(s)))
    # Joint: sweep alpha x smooth
    for a in args.alpha:
        for s in args.smooth:
            jobs.append(('dice+nll', float(a), float(s)))

    print(f"Planned runs: {len(jobs)}")
    for (loss, a, s) in jobs:
        tag = f"loss={loss}" + (f",alpha={a}" if a is not None else '') + (f",smooth={s}" if s is not None else '')
        print(f"\n=== Running {tag} ===")
        # Retry logic for OOM/resource errors by reducing batch size
        bs = int(args.batch_size)
        attempts = 0
        while True:
            try:
                f1 = run_one(
                    train_file=args.train_file,
                    eval_file=args.eval_file,
                    loss_type=loss,
                    alpha=a,
                    smooth=s,
                    batch_size=bs,
                    max_seq_len=args.max_seq_len,
                    max_epoch=args.max_epoch,
                    early_stop=early_stop,
                    patience=args.patience,
                    embeddings_name=args.embeddings,
                    csv_path=args.csv_path,
                )
                print(f"Completed {tag} micro-F1={f1*100.0:.2f} (batch_size={bs})")
                break
            except Exception as e:
                msg = str(e).lower()
                print(f"ERROR in {tag} (batch_size={bs}): {e}")
                if any(k in msg for k in ["resource exhausted", "out of memory", "oom", "memory error"]) and bs > 8 and attempts < 2:
                    bs = max(8, bs // 2)
                    attempts += 1
                    print(f"Retrying {tag} with reduced batch_size={bs}...")
                    continue
                break


if __name__ == '__main__':
    main()