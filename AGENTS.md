# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

ACMN-Pytorch is a PyTorch research codebase for Visual Question Answering (VQA) on the CLEVR dataset, implementing the CVPR 2018 paper "Visual Question Reasoning on General Dependency Tree." It is a single-service Python ML project with no web services, databases, or Docker.

### Key setup notes

- **Missing `__init__.py` files**: The `vqa_lab/` package directories are missing `__init__.py` files (they were never committed in the repo restructure). Empty `__init__.py` files must exist in: `vqa_lab/`, `vqa_lab/data/`, `vqa_lab/model/`, `vqa_lab/model/model_zoo/`, `vqa_lab/model/model_zoo/restree/`, `vqa_lab/model/model_zoo/resblocknet/`.
- **Typo in requirements filename**: The file is named `requrements.txt` (missing an "i"). Do not use it directly — it pins PyTorch 0.3 for CUDA 8.0 which is incompatible with modern Python. Install dependencies individually with modern versions.
- **`matplotlib` is an undocumented dependency** required by `vqa_lab/display.py` but not listed in the README or `requrements.txt`.

### Running the code

- All scripts expect to be run from the workspace root with `sys.path` including the root (scripts add `../` to path).
- The `restree` (ACMN) model has hardcoded `.cuda()` calls in `tree_abstract_module.py` and `tree_module.py`. It **requires a GPU** to run. The `rbn` (ResBlockNet) model works on CPU via `--gpu=False`.
- Training/evaluation requires the CLEVR dataset (18 GB) and preprocessed question data — see `README.md` for download instructions.
- Standard commands for training and evaluation are documented in `README.md`.

### No linting or automated tests

This research codebase has no linting configuration (no flake8, pylint, mypy, etc.) and no automated test suite. Verification is done by running the training/evaluation scripts.
