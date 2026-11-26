"""
Central path configuration for VeriMEL.

Every script imports the five locations below, so the repository can be moved
anywhere without editing code. All values are overridable by environment
variables; the defaults assume the following layout:

    VeriMEL/                  <- this repository
      verimel/                <- core modules
      experiments/            <- paper analyses
      figures/                <- figure scripts
      paths.py
      clipmodel/              <- local CLIP ViT-B/32 weights
      third_party/
        MEL-M3EL-main/        <- clone of the M3EL baseline
      data/                   <- datasets (not shipped)
        WikiDiverse/
        WikiMEL/
        RichpediaMEL/
      work/                   <- caches, checkpoints, logs (git-ignored)

Environment variables
---------------------
VERIMEL_DATA   dataset root, the directory that contains WikiDiverse/, WikiMEL/
               and RichpediaMEL/                        (default: <repo>/data)
VERIMEL_WORK   caches, checkpoints, logs and intermediate files
                                                         (default: <repo>/work)
VERIMEL_FIGS   output directory of the figure scripts  (default: <WORK>/figures)
M3EL_ROOT      clone of the MEL-M3EL baseline
                                      (default: <repo>/third_party/MEL-M3EL-main)
CLIP_DIR       local CLIP ViT-B/32 weights            (default: <repo>/clipmodel)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get('VERIMEL_DATA', os.path.join(ROOT, 'data'))
WORK_DIR = os.environ.get('VERIMEL_WORK', os.path.join(ROOT, 'work'))
FIG_DIR = os.environ.get('VERIMEL_FIGS', os.path.join(WORK_DIR, 'figures'))
M3EL_ROOT = os.environ.get('M3EL_ROOT',
                           os.path.join(ROOT, 'third_party', 'MEL-M3EL-main'))
CLIP_DIR = os.environ.get('CLIP_DIR', os.path.join(ROOT, 'clipmodel'))

for _d in (WORK_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)
