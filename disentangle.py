# train_disentangle_history.py
import os
import json
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm



class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)



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


def parse_ses(x):
    if isinstance(x, str):
        return 1 if x in ("1", "high", "High") else 0
    return int(x)


def locate_history_span(prompt):

    start = prompt.find("User Preference:")
    end = prompt.find("\nWhether the user will like")
    if start == -1 or end == -1 or end <= start:
        # fallback：### Input: ... ### Response:
        s2 = prompt.find("### Input:")
        e2 = prompt.find("### Response:")
        if s2 != -1 and e2 != -1 and e2 > s2:
            return s2 + len("### Input:"), e2
        return 0, len(prompt)
    return start, end


def tokenize_with_history_mask(tokenizer, prompt, max_length):

    hist_start, hist_end = locate_history_span(prompt)
    before = prompt[:hist_start]
    history = prompt[hist_start:hist_end]
    after = prompt[hist_end:]

    ids_before = tokenizer(before, add_special_tokens=True)["input_ids"]
    ids_hist = tokenizer(history, add_special_tokens=False)["input_ids"]
    ids_after = tokenizer(after, add_special_tokens=False)["input_ids"]

    input_ids = ids_before + ids_hist + ids_after
    history_mask = [0] * len(ids_before) + [1] * len(ids_hist) + [0] * len(ids_after)


    input_ids = input_ids[:max_length]
    history_mask = history_mask[:max_length]


    if sum(history_mask) == 0:
        history_mask = [1] * len(input_ids)

    return input_ids, history_mask



class MovieSESDataset(Dataset):
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
        input_ids, history_mask = tokenize_with_history_mask(
            self.tokenizer, prompt, self.max_length
        )
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "history_mask": history_mask,
            "ses_label": parse_ses(item["socioeconomic_status"]),
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


class SESDisentangledBaseLLM(nn.Module):
    def __init__(
        self,
        base_model_path,
        layer_index: int = 28,       
        ses_dim: int = 512,        
        task_dim: int = 512,         
        num_ses: int = 2,
        num_task: int = 2,
        unfreeze_base: bool = False,
        torch_dtype=None,
    ):
        super().__init__()
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.base = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype)
        self.unfreeze_base = unfreeze_base
        for p in self.base.parameters():
            p.requires_grad = unfreeze_base
        hidden_size = self.base.config.hidden_size
        self.layer_index = layer_index

        self.z_ses_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, ses_dim),
        )
        self.z_task_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, task_dim),
        )
        self.ses_classifier = nn.Linear(ses_dim, num_ses)
        self.adv_ses_classifier = nn.Sequential(
            nn.Linear(task_dim, task_dim // 2),
            nn.GELU(),
            nn.Linear(task_dim // 2, num_ses),
        )
        self.task_classifier = nn.Linear(task_dim, num_task)

    @staticmethod
    def masked_mean(hidden, history_mask):
  
        h = hidden.float()
        m = history_mask.unsqueeze(-1).float()       # [B, L, 1]
        denom = m.sum(dim=1).clamp(min=1e-6)         # [B, 1]
        return (h * m).sum(dim=1) / denom            # [B, H]


    @staticmethod
    def last_token_pool(hidden, attention_mask):

        B, L = attention_mask.shape
        positions = torch.arange(L, device=attention_mask.device).unsqueeze(0).expand(B, L)
        last_idx = (attention_mask * positions).max(dim=1).values           # [B]
        return hidden[torch.arange(B, device=hidden.device), last_idx].float()  # [B, H]

    def forward(self, input_ids, attention_mask, history_mask):
        grad_enabled = self.unfreeze_base and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[self.layer_index]      # [B, L, H]

   
        E_hist = self.masked_mean(hidden, history_mask)        # [B, H]

        E_last = self.last_token_pool(hidden, attention_mask)  # [B, H]

        z_ses  = self.z_ses_proj(E_hist)
        z_task = self.z_task_proj(E_last)

        return z_ses, z_task, E_hist, E_last  



@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    
    feats = {"E_hist": [], "E_last": [], "z_ses": [], "z_task": []}
    ses_labels, task_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        history_mask = batch["history_mask"].to(device)

     
        z_ses, z_task, E_hist, E_last = model(input_ids, attention_mask, history_mask)

        feats["E_hist"].append(E_hist.detach().cpu())
        feats["E_last"].append(E_last.detach().cpu())
        feats["z_ses"].append(z_ses.detach().cpu())
        feats["z_task"].append(z_task.detach().cpu())
        ses_labels.append(batch["ses_label"])
        task_labels.append(batch["task_label"])

    feats = {k: torch.cat(v, dim=0) for k, v in feats.items()}
    ses_labels = torch.cat(ses_labels, dim=0)
    task_labels = torch.cat(task_labels, dim=0)
    return feats, ses_labels, task_labels


def train_linear_probe(
    X_train, y_train, X_eval, y_eval, device,
    epochs=100, lr=1e-3, batch_size=256, weight_decay=1e-4,
):
    from sklearn.metrics import roc_auc_score

    d = X_train.size(1)
    probe = nn.Linear(d, 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(X_eval.to(device))
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    y_true = y_eval.cpu().numpy()
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")

    pred = (probs >= 0.5).astype(int)
    acc = float((pred == y_true).mean())
    return {"auc": float(auc), "acc": acc}


def evaluate_disentanglement(
    model, tokenizer, probe_train_path, eval_path,
    max_length, batch_size, device, probe_epochs=100, probe_lr=1e-3,
):
    probe_train_loader = DataLoader(
        MovieSESDataset(probe_train_path, tokenizer, max_length=max_length),
        batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
    )
    eval_loader = DataLoader(
        MovieSESDataset(eval_path, tokenizer, max_length=max_length),
        batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    train_feats, train_ses, train_task = extract_features(model, probe_train_loader, device)
    eval_feats, eval_ses, eval_task = extract_features(model, eval_loader, device)

    results = {}

    reps = ["E_hist", "E_last", "z_ses", "z_task"]
    for rep in reps:
        for target, ytr, yev in [("task", train_task, eval_task), ("ses", train_ses, eval_ses)]:
            m = train_linear_probe(
                train_feats[rep], ytr, eval_feats[rep], yev, device,
                epochs=probe_epochs, lr=probe_lr, batch_size=256,
            )
            results[f"{rep}_{target}_auc"] = m["auc"]
            results[f"{rep}_{target}_acc"] = m["acc"]

    print("\n[Frozen linear probe evaluation]")
    print(f"{'Representation':<15} {'Task AUC':>10} {'SES AUC':>10}")
    for rep in reps:
        print(
            f"{rep:<15} "
            f"{results[f'{rep}_task_auc']:>10.4f} "
            f"{results[f'{rep}_ses_auc']:>10.4f}"
        )
    print()
    return results



def main(
    base_model: str = "autodl-tmp/llm",
   
    train_data_path: str = "train.json",
    eval_data_path: str = "valid.json",
    probe_train_data_path: str = "",
    output_dir: str = "output_ses_disentangle_base",
    batch_size: int = 4,
    eval_batch_size: int = 8,
    grad_accum: int = 4,
    lr_heads: float = 1e-4,
    lr_base: float = 1e-5,
    num_epochs: int = 50,
    max_length: int = 512,
    layer_index: int = 4,
    ses_dim: int = 4096,
    task_dim: int = 4096,
    alpha_ses: float = 1.0,
    beta_adv: float = 0.0,
    adv_alpha: float = 1.0,
    gamma_orth: float = 0.0,
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    unfreeze_base: bool = False,
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
        unfreeze_base=unfreeze_base,
    ).to(device)

    head_params = (
        list(model.z_ses_proj.parameters())
        + list(model.z_task_proj.parameters())
        + list(model.ses_classifier.parameters())
        + list(model.adv_ses_classifier.parameters())
        + list(model.task_classifier.parameters())
    )
    param_groups = [{"params": head_params, "lr": lr_heads, "weight_decay": 0.01}]
    if unfreeze_base:
        base_params = [p for p in model.base.parameters() if p.requires_grad]
        param_groups.append({"params": base_params, "lr": lr_base, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)

    dataset = MovieSESDataset(train_data_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    total_steps = max(1, (len(dataloader) * num_epochs) // grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    optimizer.zero_grad()

    all_metrics = {}

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        running = {"loss": 0.0, "task": 0.0, "ses": 0.0, "adv": 0.0, "orth": 0.0}

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            history_mask = batch["history_mask"].to(device)
            ses_labels = batch["ses_label"].to(device)
            task_labels = batch["task_label"].to(device)

          
            z_ses, z_task, _, _ = model(input_ids, attention_mask, history_mask)

           
            task_logits = model.task_classifier(z_task)
            loss_task = F.cross_entropy(task_logits, task_labels)

      
            ses_logits = model.ses_classifier(z_ses)
            loss_ses = F.cross_entropy(ses_logits, ses_labels)

          
            z_task_rev = grad_reverse(z_task, adv_alpha)
            adv_logits = model.adv_ses_classifier(z_task_rev)
            loss_adv_ce = F.cross_entropy(adv_logits, ses_labels)

          
            def decorrelation_loss(z1, z2):
                z1 = z1.float()
                z2 = z2.float()
                z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
                z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
                cross_cov = z1.T @ z2 / z1.size(0)
                return cross_cov.pow(2).mean()

            loss_orth = decorrelation_loss(z_ses, z_task)

            loss_adv = loss_adv_ce + loss_orth

            loss = (
                loss_task
                + loss_ses
                + beta_adv * loss_adv
            ) / grad_accum

            loss.backward()

            for k, v in zip(
                ["loss", "task", "ses", "adv", "orth"],
                [
                    loss.item() * grad_accum,
                    loss_task.item(),
                    loss_ses.item(),
                    loss_adv.item(),
                ],
            ):
                running[k] += v

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            n = step + 1
            pbar.set_postfix({k: f"{v / n:.4f}" for k, v in running.items()})

      
        if eval_data_path:
            _probe_train_path = probe_train_data_path or train_data_path
            model.eval()
            metrics = evaluate_disentanglement(
                model=model, tokenizer=tokenizer,
                probe_train_path=_probe_train_path, eval_path=eval_data_path,
                max_length=max_length, batch_size=eval_batch_size, device=device,
                probe_epochs=probe_epochs, probe_lr=probe_lr,
            )
            all_metrics[f"epoch_{epoch + 1}"] = metrics
            model.train()

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    if unfreeze_base:
        model.base.save_pretrained(output_dir)

    torch.save(
        {
            "z_ses_proj": model.z_ses_proj.state_dict(),
            "z_task_proj": model.z_task_proj.state_dict(),
            "ses_classifier": model.ses_classifier.state_dict(),
            "adv_ses_classifier": model.adv_ses_classifier.state_dict(),
            "task_classifier": model.task_classifier.state_dict(),
            "config": {
                "layer_index": layer_index,
                "ses_dim": ses_dim,
                "task_dim": task_dim,
                "hidden_size": model.base.config.hidden_size,
                "unfreeze_base": unfreeze_base,
            },
        },
        os.path.join(output_dir, "heads.pt"),
    )

    if all_metrics:
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"[Done] Saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
