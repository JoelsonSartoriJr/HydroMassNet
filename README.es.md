# HydroMassNet

¡Bienvenido a HydroMassNet!

HydroMassNet es un repositorio para experimentos y pipelines de modelado hidrológico (entrenamiento, inferencia y evaluación). Este README en Español explica instalación, inicio rápido y la estructura del proyecto.

- Portugués: README.pt.md
- English: README.en.md
- Español: README.es.md

Índice
- Acerca de
- Puntos clave
- Requisitos
- Instalación
- Inicio rápido
- Estructura del repositorio
- Datos y resultados
- Contribuciones
- Licencia
- Contacto

Acerca de
HydroMassNet contiene el código para entrenar modelos, generar predicciones y evaluar resultados. Scripts principales: train.py, predict.py, evaluate.py. El código fuente está en src/.

Puntos clave
- Scripts para entrenamiento y evaluación.
- Orquestación de pipelines (run_pipeline.py).
- Configuración mediante config.yaml.
- Dependencias en requirements.txt y pyproject.toml.

Requisitos
- Python 3.12 (las dependencias actuales están fijadas para >=3.12).
- Git.
- Se recomienda entorno virtual (venv) o Poetry.

Instalación (venv + pip)
1. Crear y activar venv:
   python3.12 -m venv .venv
   source .venv/bin/activate
2. Instalar dependencias:
   pip install -r requirements.txt

Instalación (Poetry)
1. poetry install

Inicio rápido
- Entrenar:
  python train.py --config config.yaml
- Predecir:
  python predict.py --config config.yaml --input data/<tu_entrada>
- Evaluar:
  python evaluate.py --predictions results/predictions.csv --targets data/targets.csv

Notas:
- Ajuste config.yaml para rutas de datos, hiperparámetros y salidas.
- Muchas dependencias pueden ser pesadas (TensorFlow, CatBoost) — se recomienda GPU.

Estructura del repositorio (resumen)
- READMEs en tres idiomas.
- LICENSE — MIT.
- config.yaml — archivo de configuración.
- data/ — conjuntos de datos.
- results/ — artefactos y salidas.
- src/ — código fuente.
- train.py, predict.py, evaluate.py — scripts principales.
- run_pipeline.py, run_color_plot.py — utilidades.

Mejoras sugeridas (priorizadas)
1. Documentación y ejemplos
   - Añadir un dataset de ejemplo o script para descargar datos públicos.
   - Incluir ejemplos de salida y formatos esperados.
2. Instalación y dependencias
   - Proveer requirements separados (dev / full).
   - Valorar soporte para Python 3.10+ o documentar la necesidad de 3.12.
   - Añadir Dockerfile para reproducibilidad.
3. CI/CD
   - Configurar GitHub Actions para linters, tests y ejecución mínima del pipeline.
4. Calidad de código
   - Organizar utilidades en src/, exponer CLI con argumentos y añadir tests.
   - Mejor manejo de errores y validaciones.
5. Datos y modelos
   - Documentar el esquema de datos esperado.
   - Incorporar versionado de modelos y guardado de artefactos.
6. Contribuciones
   - Añadir CONTRIBUTING.md y plantillas de PR/issue.
7. Seguridad
   - Asegurar que datos sensibles no estén en el repositorio (.gitignore) y documentar el manejo de datos.

Contribuir
- Abra issues para errores o mejoras.
- Haga fork, crie una rama y envie PRs con descripciones y pruebas cuando aplique.

Licencia
Licencia MIT — ver archivo LICENSE.

Contacto
Autor principal: Joelson Sartori Junior (GitHub)

Gracias por revisar HydroMassNet. ¡Las contribuciones son bienvenidas!
