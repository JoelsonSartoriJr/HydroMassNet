# HydroMassNet — Redes Neurais Bayesianas para Massa de Hidrogênio Neutro

[Português](README.pt.md) | [English](README.en.md) | [Español](README.es.md)

HydroMassNet é um pipeline de ponta a ponta para estimar a massa de hidrogênio neutro em galáxias (log10(M_HI)) usando Redes Neurais Bayesianas (BNNs). Ele cobre aquisição de dados (catálogos astronômicos), pré-processamento, engenharia de atributos, treinamento/avaliação com incerteza, comparação de modelos e geração de figuras para publicação.

- Stack principal: TensorFlow 2.x, TensorFlow Probability, scikit-learn, LightGBM
- Acesso a dados astronômicos: astroquery (VizieR, XMatch)
- Configuração reprodutível via YAML

## Estrutura do repositório

- data/: datasets brutos e processados
- results/
  - saved_models/: pesos treinados e artefatos (inclui CSVs de predições)
  - plots/: figuras geradas durante a avaliação
  - search_results/: relatórios de seleção de features e busca de hiperparâmetros
- config/
  - config.yaml: configurações do projeto (paths, dados, treino, seleção de features, campeões)
- src/
  - download_dataset.py: constrói o catálogo completo (VizieR + cross-match opcional)
  - preprocess.py: limpeza, imputação, engenharia de atributos e salvamento do CSV processado
  - train.py: treinamento genérico (bayesianos e vanilla) para experimentos
  - evaluate.py: avalia um modelo treinado e salva predições com incerteza
  - predict.py: predição de uma amostra com incerteza
  - baseline.py: baseline de regressão linear
  - feature_selection.py: exploração/seleção de features com LightGBM
  - run_full_optimization.py: estágio 1 (seleção) + estágio 2 (busca de hiperparâmetros)
  - plotting.py, publication_plots.py: utilitários de plotagem
  - models/: camadas e arquiteturas Bayesianas
  - utils/: utilitários auxiliares
- run_pipeline.py: orquestra preprocess -> train -> evaluate -> baseline -> plots
- run_champions.py: treina/avalia os modelos campeões definidos no config
- pyproject.toml / requirements.txt: dependências
- LICENSE: MIT

## Requisitos

- Python >= 3.9
- Poetry (recomendado) ou pip/venv
- Opcional: GPU com TensorFlow configurado

## Instalação

- Poetry (recomendado)
  - git clone https://github.com/joelsonsartori/HydroMassNet.git
  - cd HydroMassNet
  - poetry install

- pip
  - git clone https://github.com/joelsonsartori/HydroMassNet.git
  - cd HydroMassNet
  - python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
  - pip install -r requirements.txt

## Configuração

A configuração principal está em config/config.yaml (alguns scripts também procuram ./config.yaml quando executados na raiz). Seções principais:

- paths: raw_data, processed_data, saved_models, plots, search_results
- data_processing: divisões de validação/teste e a lista de features do dataset processado
- training: epochs, patience
- feature_selection: features candidatas, min_features, top_n_to_tune
- champion_models: features por modelo e hiperparâmetros finais (bnn, dbnn, vanilla)

## Início rápido

1) Construir o catálogo completo (download/merge via astroquery)
- poetry run python src/download_dataset.py

2) Pré-processar e engenhar features
- poetry run python src/preprocess.py

3) Treinar + avaliar tudo e gerar plots (orquestração)
- poetry run python run_pipeline.py

Isso roda bnn, dbnn e vanilla, gera CSVs de predições em results/saved_models e figuras em results/plots.

## Uso manual

- Treinar um modelo específico
  - poetry run python src/train.py --model bnn
  - poetry run python src/train.py --model dbnn
  - poetry run python src/train.py --model vanilla
  - Flags opcionais:
    - --learning_rate, --batch_size
    - --hidden_layers para bnn/vanilla (ex.: 256-128)
    - --core_layers/--head_layers para dbnn (ex.: 512-256 e 128-64)
    - --dropout para vanilla
    - --features "iMAG,e_iMAG,logMsT,..." para sobrescrever as features

- Avaliar um modelo treinado (gera {model}_predictions.csv)
  - poetry run python src/evaluate.py --model bnn|dbnn|vanilla

- Baseline
  - poetry run python src/baseline.py

- Plots
  - Gerados automaticamente por run_pipeline.py (históricos, comparação de predições e intervalos de confiança)

- Predizer uma amostra com incerteza
  - poetry run python src/predict.py --model bnn --input_values <v1 v2 ... vN> --config config/config.yaml
  - Obs.: input_values deve ter o mesmo comprimento de data_processing.features; usa scaler_x.pkl e scaler_y.pkl.

- Otimização completa (seleção de features + busca de hiperparâmetros)
  - poetry run python src/run_full_optimization.py

- Modelos campeões (treino/avaliação finais conforme config.champion_models)
  - poetry run python run_champions.py

## Dados e features

- Alvo padrão: logMHI
- Exemplo de features (do config): iMAG, e_iMAG, logMsT, logSFR22, e_logMsT, Dist, RVel, Ag, Ai, surface_brightness_proxy
- O proxy surface_brightness_proxy é derivado no pré-processamento a partir de iMAG e razão de eixos b/a
- Dados salvos em CSVs sob data/ conforme configurado

## Reprodutibilidade

- Semente global no YAML (seed: 1601)
- Pipelines guiados por configuração e pré-processamento determinístico sempre que possível

## Resultados e artefatos

- results/saved_models/: pesos (*.weights.h5), CSVs de predições, históricos
- results/plots/: figuras PNG/PDF de métricas de treino e visualização de incerteza
- results/search_results/: relatórios CSV de seleção de features e otimização

## Licença

Projeto licenciado sob MIT. Veja LICENSE para detalhes.

## Citação

Se usar o HydroMassNet em trabalhos acadêmicos, por favor cite este repositório. Um BibTeX será adicionado quando houver preprint.

## Agradecimentos

- Serviços VizieR e XMatch via astroquery
- TensorFlow Probability para camadas Bayesianas
- Bibliotecas da comunidade: scikit-learn, LightGBM, pandas, matplotlib/seaborn
