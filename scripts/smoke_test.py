from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apollo_story.config import load_yaml
from apollo_story.model import GPT, GPTConfig
from apollo_story.optimizers import build_optimizer


def main() -> None:
    config = load_yaml("configs/base.yaml")
    config = deepcopy(config)
    config["model"]["vocab_size"] = 128
    tiny = GPT(
        GPTConfig(
            vocab_size=128,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
            bias=True,
        )
    )
    batch = {
        "input_ids": torch.randint(0, 128, (2, 32)),
        "labels": torch.randint(0, 128, (2, 32)),
    }
    for name in ["sgd", "adamw", "galore", "apollo", "apollo_mini"]:
        config["optimizer"]["name"] = name
        if name == "apollo_mini":
            config["optimizer"]["rank"] = 1
        optimizer = build_optimizer(tiny, config)
        loss = tiny(**batch)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"{name}: ok")


if __name__ == "__main__":
    main()

