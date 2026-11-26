"""
CLIP cross-encoder + 两阶段重排

阶段1（召回）: M3EL 文本分支 top-K 候选（precompute_recall.py 预计算）
阶段2（精排）: CLIP text-encoder 作为 cross-encoder —— mention 与实体文本拼接过同一个编码器，
               从 EOT 池化 + head 输出匹配分，重排召回集。

用 CLIP 的好处：全程同一向量空间（与区域/图像特征一致），复用本地 clipmodel，零下载。
代价：77 token 上限（mention+entity 截断）。

训练：每个 mention 的 gold 为正、召回候选为负，BCE 微调 CLIP 文本编码器 + 打分头。
"""
import os, sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR
RECALL_FILE = f'{WORK_DIR}/wikidiverse_recall.pt'
SAVE_PATH = f'{WORK_DIR}/cross_encoder.pt'


class CLIPCrossEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_DIR)
        self.head = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask):
        out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out[1]                                   # EOT 位置池化
        emb = self.clip.text_projection(pooled)           # [bs, 512]
        return self.head(emb).squeeze(-1)                 # [bs] 匹配分


def build_texts(entries, entity_text, k_cands=100):
    """把 mention 条目展开成 (mention_text, entity_text, label) 三元组。
    label=1 表示 gold（正），0 表示负样本。"""
    mention_strs, entity_strs, labels = [], [], []
    for e in entries:
        if e['gold'] in entity_text:
            mention_strs.append(e['mention'])
            entity_strs.append(entity_text[e['gold']])
            labels.append(1)
        cands = e['cands'][:k_cands]
        for c in cands:
            if c == e['gold']:
                continue
            mention_strs.append(e['mention'])
            entity_strs.append(entity_text.get(c, ''))
            labels.append(0)
    return mention_strs, entity_strs, np.array(labels, dtype=np.float32)


def train_cross_encoder(epochs=3, lr=1e-5, batch_mentions=8, k_cands=100, device='cuda:0', seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    recall = torch.load(RECALL_FILE, map_location='cpu')
    train_entries = recall['train']
    dev_entries = recall['dev']
    entity_text = recall['entity_text']

    tokenizer = CLIPProcessor.from_pretrained(CLIP_DIR).tokenizer
    model = CLIPCrossEncoder().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    best_dev = -1
    for ep in range(epochs):
        random.shuffle(train_entries)
        model.train()
        total, cnt = 0.0, 0
        for i in range(0, len(train_entries), batch_mentions):
            chunk = train_entries[i:i + batch_mentions]
            m_strs, e_strs, labels = build_texts(chunk, entity_text, k_cands)
            enc = tokenizer([f"{m} [SEP] {e}" for m, e in zip(m_strs, e_strs)],
                            padding='max_length', max_length=77, truncation=True, return_tensors='pt')
            scores = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            loss = F.binary_cross_entropy_with_logits(scores, torch.tensor(labels, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); cnt += 1
            if cnt % 50 == 0:
                print(f'[train] ep{ep} step{cnt} loss={total/cnt:.4f}', flush=True)
        # dev 评估（真实两阶段 rank：gold 不在召回池 = 失败）
        dev_mrr, _ = eval_ranks(model, dev_entries, entity_text, tokenizer, device, k_cands)
        print(f'[dev] ep{ep} MRR={dev_mrr:.4f}', flush=True)
        if dev_mrr > best_dev:
            best_dev = dev_mrr
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'[save] best dev MRR {dev_mrr:.4f}', flush=True)
    return model


def eval_ranks(model, entries, entity_text, tokenizer, device='cuda:0', k_cands=100):
    """两阶段评测：gold 不在召回池的 mention 直接记失败（rank=K+1），
    避免把"gold 强行放进打分池"导致的虚高。"""
    model.eval()
    ranks = []
    with torch.no_grad():
        for e in entries:
            gold, m = e['gold'], e['mention']
            if not e.get('gold_in_recall', True):
                ranks.append(k_cands + 1)
                continue
            cands = e['cands'][:k_cands]
            texts = [f"{m} [SEP] {entity_text.get(c,'')}" for c in cands]
            texts.append(f"{m} [SEP] {entity_text[gold]}")
            enc = tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
            scores = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            ranks.append(1 + int((scores[:-1] > scores[-1]).sum()))
    ranks = np.array(ranks)
    return float((1.0 / ranks).mean()), ranks


def eval_test(device='cuda:0'):
    recall = torch.load(RECALL_FILE, map_location='cpu')
    entity_text = recall['entity_text']
    tokenizer = CLIPProcessor.from_pretrained(CLIP_DIR).tokenizer
    model = CLIPCrossEncoder().to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    for split in ['dev', 'test']:
        in_cand = np.mean([bool(e.get('gold_in_recall', True)) for e in recall[split]])
        mrr, ranks = eval_ranks(model, recall[split], entity_text, tokenizer, device)
        hits = {n: (ranks <= n).mean() for n in (1, 3, 5, 10)}
        print(f'[{split}] MRR={mrr:.4f} H@1={hits[1]:.4f} H@3={hits[3]:.4f} '
              f'H@5={hits[5]:.4f} H@10={hits[10]:.4f}')
        print(f'  [recall@100] {in_cand*100:.1f}%  (召回上限，H@1 不可能超过它)')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train', choices=['train', 'eval'])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--recall', default=RECALL_FILE, help='precompute_recall.py 的产出文件')
    ap.add_argument('--save', default=SAVE_PATH, help='最优 checkpoint 保存路径')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    RECALL_FILE = args.recall
    SAVE_PATH = args.save
    dev = f'cuda:{args.gpu}'
    if args.mode == 'train':
        train_cross_encoder(epochs=args.epochs, device=dev, seed=args.seed)
        eval_test(device=dev)
    else:
        eval_test(device=dev)
