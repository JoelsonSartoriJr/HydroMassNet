# HydroMassNet

Welcome to HydroMassNet!

HydroMassNet is a repository for hydrological modeling experiments and pipelines (training, inference, evaluation). This English README provides installation steps, quick start commands and explains the project layout.

- Português: README.pt.md
- English: README.en.md
- Español: README.es.md

Contents
- About
- Highlights
- Prerequisites
- Installation
- Quick start
- Repository structure
- Data & results
- Contributing
- License
- Contact

About
HydroMassNet contains code to train models, run predictions and evaluate results. Main scripts: train.py, predict.py, evaluate.py. Project code lives under src/.

Highlights
- Training and evaluation scripts.
- Pipeline orchestration (run_pipeline.py).
- Configuration via config.yaml.
- Dependencies available in requirements.txt and pyproject.toml.

Prerequisites
- Python 3.12 (current pinned dependencies target >=3.12).
- Git.
- Use virtual environments (venv) or Poetry for reproducible installs.

Installation (venv + pip)
1. Create and activate venv:
   python3.12 -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt

Installation (Poetry)
1. poetry install

Quick start
- Train:
  python train.py --config config.yaml
- Predict:
  python predict.py --config config.yaml --input data/<your_input>
- Evaluate:
  python evaluate.py --predictions results/predictions.csv --targets data/targets.csv

Notes:
- Check and adjust config.yaml for data paths, hyperparameters and output paths.
- Many dependencies (TensorFlow, CatBoost) are heavy — GPU recommended.

Repository structure (summary)
- README files in three languages.
- LICENSE — MIT.
- config.yaml — configuration file.
- data/ — input datasets.
- results/ — outputs and artifacts.
- src/ — source code.
- train.py, predict.py, evaluate.py — main scripts.
- run_pipeline.py, run_color_plot.py — utility scripts.

Suggested improvements (prioritized)
1. Documentation & examples
   - Add a small example dataset or a script to fetch public sample data.
   - Provide example outputs and expected file formats.
2. Setup & dependencies
   - Offer lightweight dev requirements and a full requirements set.
   - Consider supporting Python 3.10+ or specify strict reason for 3.12.
   - Provide a Dockerfile for reproducibility.
3. Automation & CI
   - Add GitHub Actions for linting, basic tests and a smoke-run of the pipeline.
4. Code quality
   - Move reusable code into src/ packages, expose CLI entrypoints and add argument parsing.
   - Add unit tests for core functions.
   - Improve error handling for file I/O and configs.
5. Data & models
   - Specify data schema (expected columns and units).
   - Add model versioning and artifact saving strategy.
6. Contribution process
   - Add CONTRIBUTING.md and templates for issues/PRs.
7. Security
   - Ensure .gitignore filters datasets/credentials and document data handling.

Contributing
- Open issues for bugs or feature requests.
- Fork, create a branch and open PRs. Include tests and documentation updates.

License
MIT License — see LICENSE file.

Contact
Repository owner / primary contact: Joelson Sartori Junior (GitHub)
