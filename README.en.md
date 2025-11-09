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
- Install dependencies (venv or Poetry) and ensure the data files in `data/` are available.
- Choose the dataset you want to run:

  1. **HydroMassNet processed CSV**
     ```bash
     poetry run python main_hydromass.py --config config.yaml
     ```
     The script will preprocess (if needed), train all configured models, evaluate them and drop results under `results/`.

  2. **xGASS representative sample (FITS)**
     ```bash
     poetry run python main_xgass.py --config config_xgass.yaml
     ```
     This entrypoint loads `data/xGASS_representative_sample.fits`, creates `data/xgass_processed.csv`, trains/evaluates the models defined in `config_xgass.yaml` and generates EDA plots in `results/xgass/plots`.
     Before the final training it also runs a shared hyperparameter grid-search (same set of learning rates, batch sizes and layer sizes for every model) and stores the optimized configuration under `results/xgass/config_optimized.yaml`. Tweak the `hyperparameter_search` block inside `config_xgass.yaml` to change the number of trials or output paths.

- Optional stand-alone commands:
  - Retrain a specific model: `poetry run python train.py --model vanilla --config config_xgass.yaml`
  - Evaluate only: `poetry run python evaluate.py --model vanilla --config config_xgass.yaml`
  - Make a single prediction (after training): `poetry run python predict.py --model vanilla --input_values …`

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
