import argparse, os, random, torch
from functools import partial
from torch.utils.data import DataLoader
from rich import print as rprint

os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
from transformer_lens.model_bridge import TransformerBridge

from tasks.registry import get_task
from data.PairedDataset import PairedDataset
from data.PairedCollator import PairedCollator
from data.CountPairs import shift_range
from mechanistic.utilities.mechinterp_dataclasses import PatchingRunConfig
from mechanistic.utilities.mechinterp_utils import load_config

START_SLOT = 1
PRED_POS = 3
SITES = [
    "blocks.0.hook_resid_pre",  # after embedding
    "blocks.0.hook_resid_post",  # after layer 0
    "blocks.1.hook_resid_post",  # after layer 1
]

# Patch hook: overwrite ONE position with the cached clean activation
def patch_at(act, hook, *, clean_act, position):
    act[:, position, :] = clean_act[:, position, :]
    return act

def logit_diff(logits, clean_start_ids, corrupted_start_ids):
    row = logits[:, PRED_POS, :]
    return (row.gather(1, clean_start_ids.unsqueeze(1)) - row.gather(1, corrupted_start_ids.unsqueeze(1))).squeeze(1).mean().item()

def main():
    # Read configuration
    parser = argparse.ArgumentParser(description="Run pruning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to the config file'
    )
    args = parser.parse_args()
    run_config: PatchingRunConfig = load_config(args.config)
    
    # Load the model - linearize LayerNorms, center unembedding, center writing
    # weights and factor attention matrices using SVD
    model = TransformerBridge.boot_transformers(
        run_config.model_path
    )
    model.enable_compatibility_mode(
        fold_ln=True,
        center_unembed=True,
        center_writing_weights=True,
        refactor_factored_attn_matrices=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Initialize task tokenizers, datasets and collators
    task_config = run_config.task_config
    task = get_task(task_config.name, task_config)
    tokenizer, datasets = task.build()
    paired = PairedDataset(datasets['train'], shift_range)
    loader = DataLoader(
        paired,
        batch_size=run_config.batch_size,
        collate_fn=PairedCollator(tokenizer.pad_token_id)
    )
    
    batch = next(iter(loader))
    clean = {k: v.to(run_config.torch_device) for k, v in batch['clean'].items()}
    corrupted = {k: v.to(run_config.torch_device) for k, v in batch['corrupted'].items()}
    
    rprint(tokenizer.convert_ids_to_tokens(clean['input_ids'][0]))
    rprint(clean['position_ids'][0])
    
    # Cache clean residuals + clean baseline logits in one pass
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(
            clean['input_ids'], name_filter=SITES,
            position_ids=clean['position_ids']
        )
        corrupted_logits = model(
            corrupted['input_ids'],
            position_ids=corrupted['position_ids']
        )

    patched = {}
    with torch.no_grad():
        for site in SITES:
            patched[site] = model.run_with_hooks(
                corrupted['input_ids'],
                fwd_hooks=[(site, partial(patch_at, clean_act=cache[site], position=START_SLOT))],
                position_ids=corrupted['position_ids'],
            )
    
    # Logit-diff metric at PRED_POS (predicting body[0])
    clean_start_ids = clean['input_ids'][:, START_SLOT]
    corrupted_start_ids = corrupted['input_ids'][:, START_SLOT]
    
    ld_clean = logit_diff(clean_logits, clean_start_ids, corrupted_start_ids)
    ld_corrupted = logit_diff(corrupted_logits, clean_start_ids, corrupted_start_ids)
    span = ld_clean - ld_corrupted
    
    print(f"Clean        : {ld_clean:+.3f}")
    print(f"Corrupted    : {ld_corrupted:+.3f}")
    for site in SITES:
        ld = logit_diff(patched[site], clean_start_ids, corrupted_start_ids)
        rec = (ld - ld_corrupted) / span if span else float('nan')
        print(f"   Patched @ {site:30s} {ld:+.3f} Recovery: {rec*100:5.1f}%")

if __name__ == "__main__":
    main()