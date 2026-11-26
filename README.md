# VeriMEL: Verified Visual Evidence for Multimodal Entity Linking

VeriMEL is a two-stage framework for multimodal entity linking. Full-corpus
matching is decoupled into text-only retrieval and a text cross-encoder
re-ranker; visual information is demoted to evidence that is only added when
verified:

```
s = s_text + lambda * 1[v > tau] * v
```

where `v` is the CLIP cosine similarity between the mention image and the
candidate entity image, `tau` is a confidence threshold calibrated on the
development set, and `lambda` bounds the size of the correction. When no
reliable evidence exists the gate closes and the method degrades exactly to its
text-only floor.

On WikiDiverse, WikiMEL and RichpediaMEL this reproduces the main results of
the paper (test MRR / H@1): **83.73 / 77.86**, **93.77 / 90.91** and
**91.88 / 88.15**.

## Layout

| Path | Contents |
| --- | --- |
| `paths.py` | all locations; overridable via environment variables |
| `verimel/` | retrieval cache, cross-encoder re-ranker, evidence gate |
| `third_party/` | how to obtain the M3EL baseline code and checkpoints |

## Setup

```bash
pip install -r requirements.txt
```

1. **CLIP weights** (ViT-B/32):

   ```bash
   python -c "from transformers import CLIPModel; \
       CLIPModel.from_pretrained('openai/clip-vit-base-patch32').save_pretrained('clipmodel')"
   ```

2. **M3EL baseline** (retrieval stage). Clone it into `third_party/` and use
   its released text-branch checkpoints — see
   [`third_party/README.md`](third_party/README.md).

3. **Datasets** (WikiDiverse / WikiMEL / RichpediaMEL). Place them under
   `data/` or point `VERIMEL_DATA` at them; each directory holds the split
   JSONs, `kb_entity.json`, `kb_image/`, `mention_image/`, `qid2id.json`.
   All paths default to this layout and can be overridden:

   ```bash
   export VERIMEL_DATA=/path/to/data      # datasets
   export VERIMEL_WORK=/path/to/work      # caches, checkpoints, logs
   export M3EL_ROOT=/path/to/MEL-M3EL-main
   export CLIP_DIR=/path/to/clipmodel
   ```

## Pipeline

Substitute `wikidiverse` / `wikimel` for the other benchmarks.

```bash
DS=richpediamel
CFG=$M3EL_ROOT/config/$DS.yaml
CKPT=$M3EL_ROOT/runs/RichpediaMEL/version_0/checkpoints/epoch=11-step=1176.ckpt

# 1. entity image embeddings (CLIP image tower, cached)
python verimel/precompute_entity_img_emb.py --config $CFG

# 2. stage-1 retrieval: M3EL text branch, top-100 per mention
python verimel/precompute_recall.py --config $CFG --ckpt $CKPT --topk 100

# 3. mention image embeddings for all splits
python verimel/prepare_mention_img.py --ds $DS --split all

# 4. train the cross-encoder re-ranker (3 epochs)
python verimel/cross_encoder.py --mode train --gpu 0 \
    --recall $VERIMEL_WORK/${DS}_recall.pt \
    --save   $VERIMEL_WORK/cross_encoder_${DS}.pt

# 5. calibrate the evidence gate on dev, report test
python verimel/gated_rerank_sweep.py --ds $DS --lam-max 12.8
```

Step 5 prints the full dev grid and the test result at the dev optimum — the
latter is the paper's main number for this benchmark.
