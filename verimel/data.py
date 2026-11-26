"""
MELCODE 数据模块：复用 M3EL 的 DataModuleForM3EL，
仅覆盖 train_collator —— 在 batch 里额外注入 gold_id（每个 mention 的 gold 实体 id），
供 listwise 训练采样困难负样本时屏蔽 gold 使用。
"""
import copy
import torch
import sys, os

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # repo root: paths.py
_sys.path.insert(0, _HERE)
from paths import ROOT, DATA_ROOT, M3EL_ROOT, CLIP_DIR, WORK_DIR, FIG_DIR
sys.path.insert(0, M3EL_ROOT)
from codes.utils.dataset import DataModuleForM3EL


class DataModuleForListwise(DataModuleForM3EL):
    def train_collator(self, samples):
        cls_idx, img_list, sample_type, input_dict_list = [], [], [], []
        pixel_values, gt_ent_id = [], []

        for sample_idx, sample in enumerate(samples):
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)
            gt_ent_id.append(sample.pop('answer'))

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx]))

        input_dict = self.tokenizer.pad(input_dict_list,
                                        padding='max_length',
                                        max_length=self.args.data.text_max_length,
                                        return_tensors='pt')
        pixel_values = torch.stack(pixel_values)
        input_dict['pixel_values'] = pixel_values

        # 实体信息
        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in gt_ent_id]
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = [], [], [], []
        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            ent_input_dict_list.append(ent_dict)
        for idx, _ in enumerate(ent_input_dict_list):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))

        ent_empty_img_flag = torch.tensor([True if not len(_) else False for _ in ent_img_list], dtype=torch.bool)
        ent_input_dict = self.tokenizer.pad(ent_input_dict_list,
                                            padding='max_length',
                                            max_length=self.args.data.text_max_length,
                                            return_tensors='pt')
        ent_pixel_values = torch.stack(ent_pixel_values)
        ent_input_dict['pixel_values'] = ent_pixel_values
        ent_input_dict['empty_img_flag'] = ent_empty_img_flag

        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v

        # ---- listwise 额外注入 gold 实体 id ----
        input_dict['gold_id'] = torch.tensor(gt_ent_id, dtype=torch.long)
        return input_dict
