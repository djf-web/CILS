# bias_neuron_locate.py
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP



class SESDisentangledBaseLLM(nn.Module):
    def __init__(self, base_model_path, layer_index=16, ses_dim=256, task_dim=256,
                 num_ses=2, num_task=2, torch_dtype=None):
        super().__init__()
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.base = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype)
        for p in self.base.parameters():
            p.requires_grad = False
        hs = self.base.config.hidden_size
        self.layer_index = layer_index
        self.z_ses_proj = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.GELU(), nn.Linear(hs // 2, ses_dim))
        self.z_task_proj = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.GELU(), nn.Linear(hs // 2, task_dim))
        self.ses_classifier = nn.Linear(ses_dim, num_ses)
        self.adv_ses_classifier = nn.Linear(task_dim, num_ses)
        self.task_classifier = nn.Linear(task_dim, num_task)

    @staticmethod
    def masked_mean(hidden, history_mask):
        h = hidden.float()
        m = history_mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)



def generate_prompt(instruction, input_text=None):
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def locate_history_span(prompt):
    s = prompt.find("User Preference:")
    e = prompt.find("\nWhether the user will like")
    if s == -1 or e == -1 or e <= s:
        s2 = prompt.find("### Input:")
        e2 = prompt.find("### Response:")
        if s2 != -1 and e2 != -1 and e2 > s2:
            return s2 + len("### Input:"), e2
        return 0, len(prompt)
    return s, e


def tokenize_with_history_mask(tokenizer, prompt, max_length):
    hs, he = locate_history_span(prompt)
    before, history, after = prompt[:hs], prompt[hs:he], prompt[he:]
    ids_before = tokenizer(before, add_special_tokens=True)["input_ids"]
    ids_hist = tokenizer(history, add_special_tokens=False)["input_ids"]
    ids_after = tokenizer(after, add_special_tokens=False)["input_ids"]
    input_ids = ids_before + ids_hist + ids_after
    hm = [0] * len(ids_before) + [1] * len(ids_hist) + [0] * len(ids_after)
    input_ids = input_ids[:max_length]
    hm = hm[:max_length]
    if sum(hm) == 0:
        hm = [1] * len(input_ids)
    return input_ids, hm


class MovieSESDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = generate_prompt(item["instruction"], item["input"])
        input_ids, hm = tokenize_with_history_mask(self.tokenizer, prompt, self.max_length)
        ses = item["socioeconomic_status"]
        if isinstance(ses, str):
            ses_label = 1 if ses in ("1", "高", "high", "High") else 0
        else:
            ses_label = int(ses)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "history_mask": hm,
            "ses_label": ses_label,
            "task_label": 1 if item["output"].strip() == "Yes." else 0,
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, history_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append([pad_token_id] * pad_len + x["input_ids"])
        attention_mask.append([0] * pad_len + x["attention_mask"])
        history_mask.append([0] * pad_len + x["history_mask"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "history_mask": torch.tensor(history_mask, dtype=torch.long),
        "ses_label": torch.tensor([x["ses_label"] for x in batch], dtype=torch.long),
        "task_label": torch.tensor([x["task_label"] for x in batch], dtype=torch.long),
    }



class MLPActivationCapture:
    def __init__(self, base_model, alpha=1.0):
        self.base_model = base_model
        self.alpha = alpha
        self.orig_forwards = []
        self.raw_acts = []

    def __enter__(self):
        self.orig_forwards = []
        self.raw_acts = []
        ctx = self
        for layer in self.base_model.model.layers:
            mlp = layer.mlp
            orig = mlp.forward
            self.orig_forwards.append(orig)

            def custom_forward(self_mlp, hidden_states):
                gate = self_mlp.gate_proj(hidden_states)
                up = self_mlp.up_proj(hidden_states)
                raw = F.silu(gate.float()) * up.float()        # [B, L, H_inter]
                ctx.raw_acts.append(raw)
                scaled = (raw * ctx.alpha).to(hidden_states.dtype)
                return self_mlp.down_proj(scaled)

            mlp.forward = custom_forward.__get__(mlp, LlamaMLP)
        return self

    def __exit__(self, *args):
        for layer, orig in zip(self.base_model.model.layers, self.orig_forwards):
            layer.mlp.forward = orig
        self.orig_forwards.clear()



def train_E_task_probe(model, tokenizer, data_path, device,
                       max_length=512, epochs=200, lr=1e-3, batch_size=256):

    ds = MovieSESDataset(data_path, tokenizer, max_length=max_length)
    dl = DataLoader(ds, batch_size=8, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    Es, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Collect E features"):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            hm = batch["history_mask"].to(device)
            outs = model.base(input_ids=ids, attention_mask=am,
                              output_hidden_states=True, return_dict=True)
            h = outs.hidden_states[model.layer_index]
            E = model.masked_mean(h, hm).cpu()
            Es.append(E)
            ys.append(batch["task_label"])

    E_all = torch.cat(Es, 0)
    y_all = torch.cat(ys, 0)
    print(f"[E probe] E shape = {E_all.shape}, pos_rate = {y_all.float().mean().item():.4f}")

    probe = nn.Linear(E_all.size(1), 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    ds2 = TensorDataset(E_all, y_all)
    dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in dl2:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    print("[E probe] trained")
    return probe



def locate_sample_bias_neurons(model, probe_E_task, batch, device):
    ids = batch["input_ids"].to(device)
    am = batch["attention_mask"].to(device)
    hm = batch["history_mask"].to(device)
    task_y = batch["task_label"].to(device)


    with torch.no_grad():
        embeds = model.base.get_input_embeddings()(ids)
    embeds = embeds.detach().requires_grad_(True)

    with MLPActivationCapture(model.base, alpha=1.0) as cap:
        outs = model.base(
            inputs_embeds=embeds,   
            attention_mask=am,
            output_hidden_states=True,
            return_dict=True,
        )
        h = outs.hidden_states[model.layer_index]
        E = model.masked_mean(h, hm)                
        z_task = model.z_task_proj(E)


        logits_E = probe_E_task(E)
        loss_E = F.cross_entropy(logits_E, task_y)

    
        logits_z = model.task_classifier(z_task)
        loss_z = F.cross_entropy(logits_z, task_y)

     
        if not cap.raw_acts[0].requires_grad:
            raise RuntimeError("raw_acts no grad_fn")

        grads_E = torch.autograd.grad(
            loss_E, cap.raw_acts,
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        grads_z = torch.autograd.grad(
            loss_z, cap.raw_acts,
            retain_graph=False, create_graph=False, allow_unused=True,
        )

        acts_cpu = [a.detach().float().cpu() for a in cap.raw_acts]
        gE_cpu = [
            (g.detach().float().cpu() if g is not None else torch.zeros_like(a))
            for g, a in zip(grads_E, acts_cpu)
        ]
        gZ_cpu = [
            (g.detach().float().cpu() if g is not None else torch.zeros_like(a))
            for g, a in zip(grads_z, acts_cpu)
        ]

    per_layer_E, per_layer_Z = [], []
    for a, gE, gZ in zip(acts_cpu, gE_cpu, gZ_cpu):
        per_layer_E.append((a.abs() * gE.abs()).mean(dim=(0, 1)))
        per_layer_Z.append((a.abs() * gZ.abs()).mean(dim=(0, 1)))
    return per_layer_E, per_layer_Z



def compute_global_bias_neurons(model, probe_E_task, tokenizer, data_path, device,
                                max_length=512, max_samples=500, save_path=None):
    ds = MovieSESDataset(data_path, tokenizer, max_length=max_length)
    n = min(len(ds), max_samples)
    L = model.base.config.num_hidden_layers
    H = model.base.config.intermediate_size

    sum_E = [torch.zeros(H) for _ in range(L)]
    sum_Z = [torch.zeros(H) for _ in range(L)]
    valid = 0

    for i in tqdm(range(n), desc="Locating bias neurons"):
        batch = collate_fn([ds[i]], tokenizer.pad_token_id)
        try:
            pE, pZ = locate_sample_bias_neurons(model, probe_E_task, batch, device)
            for l in range(L):
                sum_E[l] += pE[l]
                sum_Z[l] += pZ[l]
            valid += 1
        except Exception as e:
            print(f"[skip {i}] {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[Valid] {valid}/{n}")
    if valid == 0:
        raise RuntimeError("No valid samples")

    mean_E = torch.stack([s / valid for s in sum_E])   # [L, H_inter]
    mean_Z = torch.stack([s / valid for s in sum_Z])   # [L, H_inter]


    bias_score = mean_E - mean_Z
    ratio = mean_E / (mean_Z + 1e-8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "mean_E": mean_E,
            "mean_Z": mean_Z,
            "bias_score": bias_score,
            "ratio": ratio,
        }, save_path)
        print(f"[Saved] {save_path}")


    flat = bias_score.flatten()
    top_vals, top_ids = torch.topk(flat, 20)
    print("\nTop-20：")
    for r, (idx, v) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), 1):
        l = idx // H
        nid = idx % H
        print(f"  #{r:2d}: layer={l:2d} neuron={nid:5d}  "
              f"E={mean_E[l, nid]:.4e}  "
              f"z_task={mean_Z[l, nid]:.4e}  "
              f"bias={v:.4e}")

    return {"mean_E": mean_E, "mean_Z": mean_Z, "bias_score": bias_score, "ratio": ratio}



def main(
    base_model: str = "autodl-tmp/llm",
    ckpt_dir: str = "output_ses_disentangle_base",
    data_path: str = "train1.json",
    save_path: str = "bias_neuron_index/z.pt",
    max_length: int = 512,
    max_samples: int = 500,
    layer_index: int = 16,
    ses_dim: int = 4096,
    task_dim: int = 4096,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = SESDisentangledBaseLLM(
        base_model_path=base_model,
        layer_index=layer_index,
        ses_dim=ses_dim,
        task_dim=task_dim,
    ).to(device)

    ckpt = torch.load(os.path.join(ckpt_dir, "heads.pt"), map_location=device)
    model.z_ses_proj.load_state_dict(ckpt["z_ses_proj"])
    model.z_task_proj.load_state_dict(ckpt["z_task_proj"])
    model.ses_classifier.load_state_dict(ckpt["ses_classifier"])
    model.adv_ses_classifier.load_state_dict(ckpt["adv_ses_classifier"])
    model.task_classifier.load_state_dict(ckpt["task_classifier"])
    model.eval()
    print("[Loaded] disentangler")


    probe_E_task = train_E_task_probe(
        model, tokenizer, data_path, device,
        max_length=max_length, epochs=probe_epochs, lr=probe_lr,
    )

    compute_global_bias_neurons(
        model, probe_E_task, tokenizer, data_path, device,
        max_length=max_length, max_samples=max_samples, save_path=save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="autodl-tmp/llm")
    parser.add_argument("--ckpt_dir", type=str, default="output_ses_disentangle_base")
    parser.add_argument("--data_path", type=str, default="train.json")
    parser.add_argument("--save_path", type=str,
                        default="bias_neuron_index/z.pt")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--layer_index", type=int, default=16)
    parser.add_argument("--ses_dim", type=int, default=4096)
    parser.add_argument("--task_dim", type=int, default=4096)
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(**vars(args))