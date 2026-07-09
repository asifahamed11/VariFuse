import os
import logging
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from config import STAGE08_OUT, STAGE09_OUT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("stage10_esm")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("TORCH_HOME", str(STAGE09_OUT / "torch_cache"))
if _DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
LAYER = 33
EMBED_DIM = 1280
MAX_LEN = 2500
WINDOW_SIZE = 1022
MASKED = False
MAX_BATCH_TOKENS = 16000
MAX_BATCH_PROTEINS = 32
USE_FP16 = True


def extract_features(input_csv, output_csv, cache_file, model, alphabet, batch_converter, mask_idx):
    logger.info("Loading dataset: %s", input_csv)
    if not input_csv.exists():
        logger.warning("File not found: %s. Skipping.", input_csv)
        return
    df = pd.read_csv(input_csv)
    required = {"protein_sequence", "aa_pos", "aa_ref", "aa_alt"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing required columns: {sorted(required - set(df.columns))}")
    N = len(df)
    E = np.zeros((N, EMBED_DIM), dtype=np.float32)
    scores = np.full(N, np.nan, dtype=np.float32)
    n_ok, n_fail = 0, 0

    # group variants by protein sequence
    short_items, long_items = [], []
    for seq, grp in df.groupby("protein_sequence"):
        seq = str(seq)
        variants = [(df.index.get_loc(idx), int(df.at[idx, "aa_pos"]),
                     str(df.at[idx, "aa_ref"]), str(df.at[idx, "aa_alt"]))
                    for idx in grp.index]
        (short_items if len(seq) <= MAX_LEN else long_items).append((seq, variants))

    def _forward_batch(seqs):
        data = [(f"p{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(_DEVICE)
        with torch.inference_mode():
            out = model(tokens, repr_layers=[LAYER])
        reps = out["representations"][LAYER]
        logps = torch.log_softmax(out["logits"].float(), dim=-1)
        return reps, logps

    def _fill(rep, logp, variants):
        nonlocal n_ok, n_fail
        for row, pos, wt, mt in variants:
            try:
                E[row] = rep[pos].detach().cpu().numpy().astype(np.float32)
                lp = logp[pos]
                scores[row] = float(lp[alphabet.get_idx(mt)] - lp[alphabet.get_idx(wt)])
                n_ok += 1
            except Exception:
                n_fail += 1

    def _masked_fill(seq, variants):
        nonlocal n_ok, n_fail
        _, _, base = batch_converter([("p", seq)])
        base = base.to(_DEVICE)
        with torch.inference_mode():
            bout = model(base, repr_layers=[LAYER])
        rep = bout["representations"][LAYER][0]
        for row, pos, wt, mt in variants:
            try:
                E[row] = rep[pos].detach().cpu().numpy().astype(np.float32)
                masked = base.clone(); masked[0, pos] = mask_idx
                with torch.inference_mode():
                    o2 = model(masked)
                lp = torch.log_softmax(o2["logits"][0].float(), dim=-1)[pos]
                scores[row] = float(lp[alphabet.get_idx(mt)] - lp[alphabet.get_idx(wt)])
                n_ok += 1
            except Exception:
                n_fail += 1

    # ---- short proteins: dynamic batching, ONE pass per protein ----
    short_items.sort(key=lambda x: len(x[0]))
    pbar = tqdm(total=len(short_items), desc=f"ESM proteins ({input_csv.name})")
    i = 0
    while i < len(short_items):
        batch, tok = [], 0
        while i < len(short_items) and len(batch) < MAX_BATCH_PROTEINS:
            cost = len(short_items[i][0]) + 2
            if batch and tok + cost > MAX_BATCH_TOKENS:
                break
            batch.append(short_items[i]); tok += cost; i += 1
        if MASKED:
            for seq, variants in batch:
                _masked_fill(seq, variants)
            pbar.update(len(batch))
            continue
        try:
            reps, logps = _forward_batch([b[0] for b in batch])
            for j, (seq, variants) in enumerate(batch):
                _fill(reps[j], logps[j], variants)
            del reps, logps
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                for seq, variants in batch:  # retry one at a time
                    try:
                        r, l = _forward_batch([seq]); _fill(r[0], l[0], variants)
                    except Exception:
                        n_fail += len(variants)
            else:
                raise
        pbar.update(len(batch))
    pbar.close()

    # ---- long proteins: window per variant ----
    for seq, variants in tqdm(long_items, desc="ESM long proteins"):
        for row, pos, wt, mt in variants:
            try:
                start = max(0, pos - 1 - WINDOW_SIZE // 2)
                end = min(len(seq), start + WINDOW_SIZE)
                if end == len(seq):
                    start = max(0, end - WINDOW_SIZE)
                wseq, wpos = seq[start:end], pos - start
                _, _, tokens = batch_converter([("p", wseq)])
                tokens = tokens.to(_DEVICE)
                with torch.inference_mode():
                    out = model(tokens, repr_layers=[LAYER])
                rep = out["representations"][LAYER][0]
                if MASKED:
                    masked = tokens.clone(); masked[0, wpos] = mask_idx
                    with torch.inference_mode():
                        out = model(masked)
                logp = torch.log_softmax(out["logits"][0].float(), dim=-1)
                E[row] = rep[wpos].detach().cpu().numpy().astype(np.float32)
                lp = logp[wpos]
                scores[row] = float(lp[alphabet.get_idx(mt)] - lp[alphabet.get_idx(wt)])
                n_ok += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                n_fail += 1
            except Exception:
                n_fail += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, E)
    df["esm_variant_score"] = scores
    df.to_csv(output_csv, index=False)
    scoring = "masked-marginal" if MASKED else "wt-marginal"
    logger.info("Extraction (%s): %d ok, %d failed (%.1f%% coverage)",
                scoring, n_ok, n_fail, 100.0 * n_ok / max(N, 1))
    logger.info("Saved embeddings -> %s", cache_file)
    logger.info("Saved enriched CSV -> %s", output_csv)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["internal", "external", "both"], default="both")
    args = parser.parse_args()

    tasks = []
    if args.dataset in ["internal", "both"]:
        tasks.append({
            "in_csv": STAGE08_OUT / "Final_Dataset_Balanced_ESM.csv",
            "out_csv": STAGE09_OUT / "Final_Dataset_Balanced_with_ESM_Score.csv",
            "out_npy": STAGE09_OUT / "esm_residue_embeddings.npy",
        })
    if args.dataset in ["external", "both"]:
        from config import DATA_DIR
        tasks.append({
            "in_csv": DATA_DIR / "external" / "external_ready_for_esm.csv",
            "out_csv": DATA_DIR / "external" / "external_with_ESM_Score.csv",
            "out_npy": DATA_DIR / "external" / "external_esm_embeddings.npy",
        })
    if not tasks:
        return

    import esm
    logger.info("Loading ESM-2 (%s) on %s ...", ESM_MODEL_NAME, _DEVICE)
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    model.eval().to(_DEVICE)
    if _DEVICE == "cuda" and USE_FP16:
        model = model.half()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    for task in tasks:
        extract_features(task["in_csv"], task["out_csv"], task["out_npy"],
                         model, alphabet, batch_converter, mask_idx)
    logger.info("Done. Next: Stage 11 (python 11_train_and_evaluate.py).")


if __name__ == "__main__":
    main()