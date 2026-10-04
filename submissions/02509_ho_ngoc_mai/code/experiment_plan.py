"""Controlled experiment configurations.  Every selection is made on validation only."""
from __future__ import annotations
from train import Config


BACKBONES = [
    Config(exp_id="B01", backbone="resnet50"),
    Config(exp_id="B02", backbone="resnext50_32x4d"),
    Config(exp_id="B03", backbone="convnext_tiny"),
    Config(exp_id="B04", backbone="deit_small_patch16_224"),
    Config(exp_id="B05", backbone="mobilenetv3_large_100"),
]

# All T experiments use the backbone selected from B01--B05; replace only `backbone`
# after reading the validation table.  Each pair changes one axis from T00.
TRAINING = [
    Config(exp_id="T00"),
    Config(exp_id="T01", init="scratch"), Config(exp_id="T02", init="frozen"),
    Config(exp_id="T03", aug="color"), Config(exp_id="T04", aug="randaug"),
    Config(exp_id="T05", loss="ls", label_smoothing=.1), Config(exp_id="T06", loss="focal", focal_gamma=2.),
    Config(exp_id="T07", mix="cutmix", mix_alpha=1.), Config(exp_id="T08", ema_decay=.999),
]

FINAL_SEEDS = (0, 1, 2)

# Set the backbone and recipe only after inspecting B01--B05 and T00--T08 on
# validation data.  This is intentionally a template, not a fabricated winner.
FINAL_SELECTION_TEMPLATE = {
    "exp_id": "F01",
    "backbone": "SET_FROM_VALIDATION",
    "seed_values": FINAL_SEEDS,
    "selection_source": "validation only",
    "test_policy": "one full-test pass per seed after configuration lock",
}
