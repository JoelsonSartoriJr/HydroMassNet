# file: ./src/pipelines/__init__.py
"""
Pipelines utilitários para orquestrar execuções completas por dataset.
"""

from .dataset_runner import run_dataset_pipeline

__all__ = ["run_dataset_pipeline"]
