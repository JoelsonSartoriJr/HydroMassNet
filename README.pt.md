# HydroMassNet

Bem-vindo ao HydroMassNet!

HydroMassNet é um repositório para experimentos e pipelines de modelagem hidrológica (treinamento, inferência e avaliação). Este README em Português guia você pela instalação, uso rápido e aponta a estrutura do projeto.

- Português: README.pt.md
- English: README.en.md
- Español: README.es.md

Índice
- Sobre
- Destaques
- Pré-requisitos
- Instalação
- Execução rápida
- Estrutura do repositório
- Dados e resultados
- Como contribuir
- Licença
- Contato

Sobre
HydroMassNet reúne código para treinar modelos, gerar previsões e avaliar resultados. O repositório contém scripts principais (train.py, predict.py, evaluate.py), pipelines (run_pipeline.py) e utilitários em src/.

Destaques
- Scripts para treinar e avaliar modelos (train.py, evaluate.py).
- Scripts de execução e visualização (run_pipeline.py, run_color_plot.py).
- Suporte a configuração via config.yaml.
- Dependências listadas em requirements.txt e pyproject.toml.

Pré-requisitos
- Python 3.12 (as dependências atuais foram travadas para >=3.12).
- Git
- Recomenda-se uso de ambiente virtual (venv) ou Poetry.

Instalação (exemplo com venv + pip)
1. Criar e ativar venv:
   python3.12 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
2. Instalar dependências:
   pip install -r requirements.txt

Instalação (com Poetry)
1. poetry install

Execução rápida
- Treinar:
  python train.py --config config.yaml
- Prever:
  python predict.py --config config.yaml --input data/<sua_entrada>
- Avaliar:
  python evaluate.py --predictions results/predictions.csv --targets data/targets.csv

Observações:
- Verifique/edite config.yaml para caminhos de dados, hiperparâmetros e saída.
- Alguns pacotes no requirements.txt são pesados (TensorFlow, CatBoost etc.). Utilize GPU se disponível.

Estrutura do repositório (resumo)
- README.md, README.pt.md, README.en.md, README.es.md — Documentação em 3 idiomas.
- LICENSE — MIT License.
- config.yaml — Arquivo de configuração (parâmetros e caminhos).
- data/ — Dados brutos e exemplos (não acompanham dados sensíveis).
- results/ — Artefatos gerados (modelos, previsões, plots).
- src/ — Código fonte e utilitários.
- train.py — Script de treinamento.
- predict.py — Script de inferência.
- evaluate.py — Script de avaliação.
- run_pipeline.py — Orquestração de pipeline.
- run_color_plot.py — Geração de visualizações.

Recomendações e melhorias (priorizadas)
1. Documentação e exemplos
   - Adicionar um exemplo mínimo de dataset (ou um script para baixar dados públicos).
   - Incluir comandos concretos e exemplos de saída esperada (ex.: amostra de results/predictions.csv).
2. Setup e dependências
   - Fornecer um requirements leve para desenvolvimento/testing e outro para produção (full).
   - Considerar suportar Python 3.10+ ou documentar claramente a necessidade de 3.12.
   - Incluir um Dockerfile para reprodução do ambiente.
3. Automação e CI
   - Adicionar GitHub Actions para:
     - Testes unitários/linters (flake8/ruff, black).
     - Validar instalação e execução de pipeline simples.
4. Código e qualidade
   - Mover scripts utilitários para src/ e transformar scripts em entrypoints com argumentos bem definidos (usar argparse / hydra / pydantic).
   - Adicionar testes unitários básicos para funções críticas.
   - Validar e tratar erros de I/O (paths inexistentes, formatos inválidos).
5. Dados e modelos
   - Especificar formato esperado dos dados (colunas, unidades).
   - Adicionar rotina para salvar/serializar modelos e versão dos pesos (com hash/versão).
6. Licença e contribuições
   - Já existe LICENSE (MIT) — documentar no README como aceitar contribuições (CONTRIBUTING.md).
7. Segurança e privacidade
   - Garantir que dados sensíveis não sejam comitados (adicionar regras ao .gitignore e instruções).

Como contribuir
- Abra issues descrevendo bugs ou novas funcionalidades.
- Faça fork, crie branch feature/xyz e envie PRs com descrição clara e testes quando aplicável.
- Adicione documentação para qualquer mudança que impacte uso/instalação.

Licença
Projeto licenciado sob MIT License — ver arquivo LICENSE.

Contato
Autor principal: Joelson Sartori Junior (ver perfil do GitHub)

Obrigado por usar/avaliar o HydroMassNet! Contribuições são bem-vindas.
