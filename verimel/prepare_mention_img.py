"""
计算 mention 图像嵌入（CLIP 图像塔），输出与召回缓存逐位置对齐。

为什么需要它：门控的视觉证据是 cos(mention 图, 候选实体图)，所以每个被试
mention 都要有一份图像嵌入。train split 的 dataloader 会 shuffle，不能按位置
对齐 json，因此这里统一用 (mention 解码串, gold 实体 id) 作键，从对应 split 的
json 序数据里取 imgPath，再按召回缓存的顺序排好。

输入: {WORK_DIR}/{ds}_recall.pt            （先跑 precompute_recall.py）
输出: {WORK_DIR}/{ds}_mention_img_{split}.pt
      {'emb': float16 [N,512] 已归一化, 'has_img': bool [N]}

用法:
  python verimel/prepare_mention_img.py --ds richpediamel --split all
  python verimel/prepare_mention_img.py --ds wikidiverse --split test
"""
import os

import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR
_sys.path.insert(0, M3EL_ROOT)

from data import DataModuleForListwise

CFG = {'richpediamel': 'richpediamel.yaml', 'wikimel': 'wikimel.yaml',
       'wikidiverse': 'wikidiverse.yaml'}
SPLITS = {'train': 'train_data', 'dev': 'val_data', 'test': 'test_data'}


@torch.no_grad()
def encode(paths, device, batch=256):
    """去重后批量编码，返回 {path: 归一化嵌入}"""
    model = CLIPModel.from_pretrained(CLIP_DIR).to(device).eval()
    feats = CLIPProcessor.from_pretrained(CLIP_DIR).feature_extractor
    uniq = sorted({p for p in paths if p})
    print(f'[info] 待编码 {len(uniq)} 张去重图像', flush=True)
    emb = {}
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        pv = []
        for p in chunk:
            try:
                img = Image.open(p).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                pv.append(feats(img, return_tensors='pt')['pixel_values'].squeeze(0))
            except Exception:
                pv.append(torch.zeros(3, 224, 224))
        e = model.get_image_features(pixel_values=torch.stack(pv).to(device))
        e = torch.nn.functional.normalize(e, dim=-1).cpu().half()
        for p, v in zip(chunk, e):
            emb[p] = v
        print(f'  {min(i + batch, len(uniq))}/{len(uniq)}', flush=True)
    return emb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='richpediamel')
    ap.add_argument('--split', default='all', choices=['train', 'dev', 'test', 'all'])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    ds = args.ds
    splits = list(SPLITS) if args.split == 'all' else [args.split]

    cfg = OmegaConf.load(f'{M3EL_ROOT}/config/{CFG[ds]}')
    recall = torch.load(f'{WORK_DIR}/{ds}_recall.pt', map_location='cpu')
    dm = DataModuleForListwise(cfg)
    dm.setup()
    tok = dm.tokenizer
    img_dir = os.path.normpath(M3EL_ROOT + str(cfg.data.mention_img_folder))

    for split in splits:
        rows = getattr(dm, SPLITS[split])
        # (mention 解码串, gold) -> imgPath
        key2img = {}
        for r in rows:
            m = tok.decode(r['input_ids'], skip_special_tokens=True)
            img_list = r.get('img_list') or []
            key2img[(m, int(r['answer']))] = img_list[0] if img_list else None

        entries = recall[split]
        paths = []
        for e in entries:
            p = key2img.get((e['mention'], e['gold']))
            paths.append(os.path.join(img_dir, p) if p else None)
        hit = sum(p is not None for p in paths)
        print(f'[{split}] {len(entries)} 条, 对齐到图像 {hit} '
              f'({hit / len(entries) * 100:.1f}%)', flush=True)

        emb = encode(paths, args.device)
        zero = torch.zeros(512, dtype=torch.float16)
        embs = torch.stack([emb.get(p, zero) if p else zero for p in paths])
        has = torch.tensor([p is not None and p in emb for p in paths])
        out = f'{WORK_DIR}/{ds}_mention_img_{split}.pt'
        torch.save({'emb': embs, 'has_img': has}, out)
        print(f'[done] {os.path.basename(out)}: {int(has.sum())}/{len(has)} 有图', flush=True)


if __name__ == '__main__':
    main()
