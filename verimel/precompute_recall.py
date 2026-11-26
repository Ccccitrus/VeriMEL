"""
预计算 M3EL 文本召回 top-K：对 train/dev/test 每个 mention，
用 M3EL 文本分支（TIMM，CLIP 基）对全 KB 打分，取 top-K 实体 id（排除 gold）作为：
  - cross-encoder 训练的困难负样本
  - 两阶段评估的候选集

输出: wikidiverse_recall.pt
  { 'train': [(gold_id, [cand_ids])], 'dev': [...], 'test': [...] }
  'entity_text': {id: "entity_name [SEP] desc/attr"}
"""
import os, sys, math, json
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from urllib.parse import unquote

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR
sys.path.insert(0, M3EL_ROOT)

from codes.model.lightning_m3el import LightningForM3EL
from data import DataModuleForListwise


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=f'{M3EL_ROOT}/config/wikidiverse.yaml')
    ap.add_argument('--ckpt', default=f'{M3EL_ROOT}/runs/WikiDiverse/version_0/checkpoints/epoch=15-step=1424.ckpt')
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--out', default=f'{WORK_DIR}/wikidiverse_recall.pt')
    # 大 KB 上 TIMM 的中间张量正比于 batch*chunk*77*77，默认 eval_chunk_size 会把
    # 显存顶到上限；这里允许按可用显存调小
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dev = args.device
    n_ent = int(cfg.data.num_entity)
    chunk = args.chunk or int(cfg.data.eval_chunk_size)

    model = LightningForM3EL.load_from_checkpoint(args.ckpt, map_location=dev)
    model = model.eval().to(dev)
    model.freeze()

    dm = DataModuleForListwise(cfg)

    # 实体文本映射（cross-encoder 用）
    # 注意：不能读 dm.raw_kb_entity —— setup_dataset_for_entity 会原地 pop 掉
    # entity_name/desc/attr（dataset.py:64-65），必须直接读原始 KB json，
    # 且要和 M3EL 一致地做 unquote（原始字段是 URL 编码的）
    with open(os.path.normpath(M3EL_ROOT + str(cfg.data.entity)), 'r', encoding='utf-8') as f:
        kb_raw = json.load(f)
    entity_text = {}
    for ent in kb_raw:
        name = unquote(ent.get('entity_name', ''))
        desc = unquote(ent.get('desc') or ent.get('attr') or '')
        entity_text[ent['id']] = f"{name} [SEP] {desc}"

    # 实体文本嵌入
    ent_dl = dm.entity_dataloader()
    et, ett = [], []
    with torch.no_grad():
        for batch in ent_dl:
            batch = pl.utilities.move_data_to_device(batch, dev)
            a, _, c, _ = model.encoder(**batch)
            et.append(a.cpu()); ett.append(c.cpu())
    ent_cls = torch.cat(et); ent_tok = torch.cat(ett)
    print(f'[info] 实体嵌入: {len(ent_cls)}', flush=True)

    def recall_split(dl, split_name):
        out = []
        for bi, batch in enumerate(dl):
            if 'gold_id' in batch:
                answer = batch.pop('gold_id').numpy()
            else:
                answer = batch.pop('answer').numpy()
            input_ids_cpu = batch['input_ids'].clone()   # 存 CPU 版供解码 mention 文本
            mention_part = {k: batch[k] for k in ['input_ids', 'attention_mask'] if k in batch}
            mb = pl.utilities.move_data_to_device(mention_part, dev)
            # M3ELEncoder.forward 会强制跑视觉塔（无 pixel_values 直接报错），
            # 这里绕过它直接走 CLIP 文本塔，与 encoder 返回的 text_embeds/text_seq_tokens 等价
            text_out = model.encoder.clip.text_model(
                input_ids=mb['input_ids'], attention_mask=mb.get('attention_mask'))
            mt = model.encoder.clip.text_projection(text_out[1])
            mtt = text_out[0]
            scores = []
            for idx in range(math.ceil(n_ent / chunk)):
                s, e = idx * chunk, min((idx + 1) * chunk, n_ent)
                ec = model.matcher.text_cls_layernorm(ent_cls[s:e].to(dev))
                etk = model.matcher.text_tokens_layernorm(ent_tok[s:e].to(dev))
                mc = model.matcher.text_cls_layernorm(mt)
                mtk = model.matcher.text_tokens_layernorm(mtt)
                sc = model.matcher.timm(ec, etk, mc, mtk)
                scores.append(sc)
            scores = torch.cat(scores, dim=-1)
            for i in range(len(answer)):
                gold = int(answer[i])
                s_i = scores[i].clone()
                # gold 越出 num_entity（RichpediaMEL 有 2 条，id 160933/160934，
                # M3EL 协议下这些 mention 永远无法连对）→ 记召回失败
                gold_in = False
                if gold < n_ent:
                    s_i[gold] = -1e9
                topk = s_i.topk(args.topk, dim=-1).indices.tolist()
                if gold < n_ent:
                    # gold 是否本就在召回 top-K 内（scores[i] 是未屏蔽的原始分）：
                    # 决定两阶段评测里该条是否可能成功（不在池内 = 系统必失败）
                    gold_in = bool(scores[i][gold] >= scores[i][topk[-1]])
                mention_str = dm.tokenizer.decode(input_ids_cpu[i], skip_special_tokens=True)
                # 额外保存分数（供 K 敏感性分析精确判定 gold 是否落在 top-K）
                gold_score = float(scores[i][gold]) if gold < n_ent else -1e9
                cand_scores = [float(scores[i][t]) for t in topk]
                out.append({'gold': gold, 'cands': topk,
                            'gold_in_recall': gold_in, 'mention': mention_str,
                            'gold_score': gold_score, 'cand_scores': cand_scores})
            if (bi + 1) % 50 == 0:
                print(f'[info] {split_name} {bi+1}/{len(dl)}', flush=True)
        return out

    result = {
        'train': recall_split(dm.train_dataloader(), 'train'),
        'dev': recall_split(dm.val_dataloader(), 'dev'),
        'test': recall_split(dm.test_dataloader(), 'test'),
        'entity_text': entity_text,
    }
    torch.save(result, args.out)
    print(f'[done] 保存到 {args.out}', flush=True)


if __name__ == '__main__':
    main()
