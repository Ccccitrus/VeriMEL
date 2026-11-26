"""
预计算 KB 实体图像的 CLIP 图像嵌入（有界视觉验证的基建）。

对每个实体取 image_list[0]（与 M3EL choose_image 的 is_eval 行为一致），
CLIP 视觉塔 + 投影 → 512 维 image_embeds（与文本嵌入同处 CLIP 联合空间，
可直接与 mention 文本嵌入算余弦相似度）。

输出: {id: {'emb': float16[512], 'has_img': bool}}
无图/文件缺失/损坏的实体: has_img=False, emb=0
"""
import os, sys, json
import torch
from PIL import Image
from omegaconf import OmegaConf
from urllib.parse import unquote
from transformers import CLIPModel, CLIPProcessor

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=f'{M3EL_ROOT}/config/richpediamel.yaml')
    ap.add_argument('--out', default=f'{WORK_DIR}/richpediamel_entity_img_emb.pt')
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    kb_path = os.path.normpath(M3EL_ROOT + str(cfg.data.entity))
    img_dir = os.path.normpath(M3EL_ROOT + str(cfg.data.kb_img_folder))

    with open(kb_path, encoding='utf-8') as f:
        kb = json.load(f)
    print(f'[info] KB {len(kb)} 实体, 图像目录 {img_dir}', flush=True)

    device = 'cuda:0'
    model = CLIPModel.from_pretrained(f'{CLIP_DIR}').to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    feats = CLIPProcessor.from_pretrained(f'{CLIP_DIR}').feature_extractor

    out = {}
    todo = []  # (id, img_path)
    for ent in kb:
        img_list = ent.get('image_list') or []
        if img_list:
            todo.append((ent['id'], os.path.join(img_dir, img_list[0])))
        else:
            out[ent['id']] = {'emb': torch.zeros(512, dtype=torch.float16), 'has_img': False}
    print(f'[info] 有图实体 {len(todo)}, 无图 {len(out)}', flush=True)

    def load_batch(paths):
        pix = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                pix.append(feats(img, return_tensors='pt')['pixel_values'].squeeze(0))
            except Exception:
                pix.append(None)
        return pix

    done = 0
    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            pix = load_batch([p for _, p in chunk])
            valid_idx = [j for j, v in enumerate(pix) if v is not None]
            embs = [None] * len(chunk)
            if valid_idx:
                pv = torch.stack([pix[j] for j in valid_idx]).to(device)
                img_embeds = model.get_image_features(pixel_values=pv)  # [B,512] 联合空间
                img_embeds = torch.nn.functional.normalize(img_embeds, dim=-1).cpu().half()
                for j, e in zip(valid_idx, img_embeds):
                    embs[j] = e
            for (eid, _), e in zip(chunk, embs):
                if e is None:
                    out[eid] = {'emb': torch.zeros(512, dtype=torch.float16), 'has_img': False}
                else:
                    out[eid] = {'emb': e, 'has_img': True}
            done += len(chunk)
            if (i // args.batch) % 50 == 0:
                print(f'[info] {done}/{len(todo)}', flush=True)

    torch.save(out, args.out)
    n_ok = sum(1 for v in out.values() if v['has_img'])
    print(f'[done] {n_ok}/{len(out)} 实体有嵌入 → {args.out}', flush=True)


if __name__ == '__main__':
    main()
