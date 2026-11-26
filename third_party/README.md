# Third-party code

The retrieval stage of VeriMEL reuses the text branch of **M3EL**, and several
scripts import modules from that repository (`codes.utils.dataset`,
`codes.model.lightning_m3el`, `codes.model.lightning_mimic`,
`codes.utils.contrastive_loss`, `codes.utils.functions`). That code is **not**
vendored here; clone the original repository instead and place it so that

```
third_party/MEL-M3EL-main/codes/...
```

exists, or point the `M3EL_ROOT` environment variable at your clone.

```bash
mkdir -p third_party
git clone <M3EL repository URL> third_party/MEL-M3EL-main
```

The paper also uses the checkpoints released with M3EL. Their expected layout is

```
third_party/MEL-M3EL-main/runs/WikiDiverse/version_0/checkpoints/epoch=15-step=1424.ckpt
third_party/MEL-M3EL-main/runs/WikiMEL/version_0/checkpoints/epoch=13-step=1988.ckpt
third_party/MEL-M3EL-main/runs/RichpediaMEL/version_0/checkpoints/epoch=11-step=1176.ckpt
```

as referenced in the commands of the main README.

Please follow the license of the M3EL repository for that code and those weights.
