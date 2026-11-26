"""
零训练门控视觉验证扫参：
  s_final = s_text + λ · 1[双方有图且 s_vis>τ] · s_vis
其中 s_vis = cos(mention图像, 候选实体图像)（CLIP 联合空间）。

流程：一次性算好每个 (mention, 候选) 的 s_text 与 s_vis 存矩阵，
然后在 (λ, τ) 网格上扫参——dev 选参，test 报告。gold 不在召回池 = 直接失败（诚实协议）。
"""
import os, sys, json, math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR
MELCODE = f'{WORK_DIR}/MELCODE'
from cross_encoder import CLIPCrossEncoder

DEV = 'cuda:0'


def load_all(ds, tag=''):
    recall = torch.load(f'{WORK_DIR}/{ds}{tag}_recall.pt', map_location='cpu')
    ent_img = torch.load(f'{WORK_DIR}/{ds}_entity_img_emb.pt', map_location='cpu')
    model = CLIPCrossEncoder().to(DEV)
    model.load_state_dict(torch.load(f'{WORK_DIR}/cross_encoder_{ds}{tag}.pt', map_location=DEV))
    model.eval()
    tok = CLIPProcessor.from_pretrained(f'{CLIP_DIR}').tokenizer
    return recall, ent_img, model, tok


@torch.no_grad()
def score_split(model, tok, entries, entity_text, ent_img, men_emb, men_has):
    """返回 (s_text[N,101], s_vis[N,101], miss[N])。gold 位置=第101列。"""
    N = len(entries)
    s_text = torch.zeros(N, 101)
    s_vis = torch.zeros(N, 101)
    miss = torch.zeros(N, dtype=torch.bool)
    for i, e in enumerate(entries):
        if not e.get('gold_in_recall', True):
            miss[i] = True
            continue
        m, gold, cands = e['mention'], e['gold'], e['cands'][:100]
        texts = [f"{m} [SEP] {entity_text.get(c,'')}" for c in cands]
        texts.append(f"{m} [SEP] {entity_text[gold]}")
        enc = tok(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        s_text[i] = model(enc['input_ids'].to(DEV), enc['attention_mask'].to(DEV)).cpu()
        pool = cands + [gold]
        if men_has[i]:
            mi = men_emb[i]  # [512] normalized
            cand_embs, idxs = [], []
            for j, c in enumerate(pool):
                info = ent_img.get(c)
                if info is not None and info['has_img']:
                    cand_embs.append(info['emb'].float()); idxs.append(j)
            if cand_embs:
                ce = torch.stack(cand_embs)
                s_vis[i, idxs] = ce @ mi  # cosine（都归一化过）
        if (i + 1) % 300 == 0:
            print(f'  {i+1}/{N}', flush=True)
    return s_text, s_vis, miss


def ranks_from(s_text, s_vis, miss, lam, tau):
    """s = s_text + λ·1[s_vis>τ]·s_vis；rank = 1 + 候选中分数严格高于 gold 的个数。
    miss（gold 不在召回池）= 直接 rank 101，不参与打分。"""
    g = (s_vis > tau).float() * s_vis
    s = s_text + lam * g
    gold_score = s[:, -1]
    rank = 1 + (s[:, :-1] > gold_score.unsqueeze(1)).sum(dim=1)
    rank = rank.numpy().astype(np.float64)
    rank[miss.numpy()] = 101.0
    return rank


def metrics(rank):
    return (1.0 / rank).mean(), (rank <= 1).mean()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='richpediamel',
                    help='wikidiverse / wikimel / richpediamel')
    ap.add_argument('--lam-max', type=float, default=3.2, help='λ 网格上界')
    ap.add_argument('--refresh', action='store_true', help='强制重算打分矩阵')
    ap.add_argument('--tag', default='', help='低资源等变体的文件名后缀（如 _p10）')
    args = ap.parse_args()
    ds = args.ds
    tag = args.tag
    recall, ent_img, model, tok = load_all(ds, tag)
    entity_text = recall['entity_text']

    scores = {}
    cache = f'{WORK_DIR}/{ds}{tag}_sweep_scores.pt'
    if os.path.exists(cache) and not args.refresh:
        print(f'[info] 使用缓存的打分矩阵 {cache}', flush=True)
        scores = torch.load(cache, map_location='cpu')
        scores = {sp: (v['s_text'], v['s_vis'], v['miss']) for sp, v in scores.items()}
    else:
        for split in ['dev', 'test']:
            entries = recall[split]
            mi = torch.load(f'{WORK_DIR}/{ds}_mention_img_{split}.pt', map_location='cpu')
            men_emb, men_has = mi['emb'].float(), mi['has_img'].bool()
            print(f'[info] scoring {split} ({len(entries)})...', flush=True)
            scores[split] = score_split(model, tok, entries, entity_text, ent_img, men_emb, men_has)
        torch.save({sp: {'s_text': v[0], 's_vis': v[1], 'miss': v[2]} for sp, v in scores.items()}, cache)

    # 基线（λ=0）
    for split in ['dev', 'test']:
        r = ranks_from(*scores[split], 0.0, 0.0)
        m, h1 = metrics(r)
        print(f'[baseline λ=0] {split}: MRR={m:.4f} H@1={h1*100:.2f}%', flush=True)

    # dev 网格扫参（λ 上探到饱和）
    lams = [x for x in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8] if x <= args.lam_max]
    taus = [0.0, 0.15, 0.25, 0.35]
    best = (-1, None)
    s_text_d, s_vis_d, miss_d = scores['dev']
    for lam in lams:
        for tau in taus:
            r = ranks_from(s_text_d, s_vis_d, miss_d, lam, tau)
            m, h1 = metrics(r)
            if m > best[0]:
                best = (m, (lam, tau))
            print(f'  dev λ={lam:<4} τ={tau:<4}: MRR={m:.4f} H@1={h1*100:.2f}%', flush=True)
    lam, tau = best[1]
    print(f'[best dev] λ={lam} τ={tau} MRR={best[0]:.4f}', flush=True)

    r = ranks_from(*scores['test'], lam, tau)
    m, h1 = metrics(r)
    print(f'[test @ best] MRR={m:.4f} H@1={h1*100:.2f}%', flush=True)


if __name__ == '__main__':
    main()
