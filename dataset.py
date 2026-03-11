from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import CLIPTokenizer
import pandas as pd

def csv_to_text(s):
    s = s.lower()
    s = s.replace(".", "")
    s = s.strip()
    s = s.replace(" ", "_")
    return s

class CLIPDataset(Dataset):
    def __init__(self, data_path, instruction_csv):
        self.samples = []

        df = pd.read_csv(instruction_csv)
        
        # instruction -> (task, condition)
        self.inst2tc = {}
        for _, row in df.iterrows():
            key = csv_to_text(row["instruction"])
            self.inst2tc[key] = (int(row["reward_enum"]), int(row["condition_0"]))

        for instruction in os.listdir(data_path):
            folder_path = os.path.join(data_path, instruction)
            if not os.path.isdir(folder_path):
                continue

            task, condition = self.inst2tc[instruction]

            for fname in os.listdir(folder_path):
                if fname.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(folder_path, fname),
                         instruction,
                         task,
                         condition)
                    )
        all_max = 0
        for path, _, _, _ in self.samples:
            x = np.load(path)
            all_max = max(all_max, x.max())

        self.num_classes = int(all_max) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        np_path, text, task, condition = self.samples[idx]
        level = np.load(np_path)

        if level.ndim == 3:
            level = level[0]

        level = torch.from_numpy(level).long()
        level = F.one_hot(level, num_classes=self.num_classes).float()
        level = level.permute(2,0,1)

        return level, text, task, condition

def make_collate_fn(tokenizer_model):
    def collate_fn(batch):
        levels, texts, tasks, conditions = zip(*batch)
        levels = torch.stack(levels)
        tasks = torch.tensor(tasks)
        conditions = torch.tensor(conditions)

        clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model)

        text_inputs = clip_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "level": levels,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "task": tasks,
            "condition": conditions,
            "raw_text": list(texts),
        }
    return collate_fn