import matplotlib.pyplot as plt
import os

# Mantenha suas funções de plot existentes aqui, mas modifique-as
# para salvar as figuras em vez de (ou além de) mostrá-las.

def plot_metrics(elbo, mae, r2, model_name, save_dir):
    """Plota as métricas de ELBO, MAE e R2."""
    plt.figure(figsize=(21, 7))

    # ELBO
    plt.subplot(1, 3, 1)
    plt.plot(elbo)
    plt.title(f'{model_name} - ELBO per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')

    # MAE
    plt.subplot(1, 3, 2)
    plt.plot(mae)
    plt.title(f'{model_name} - MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')

    # R2 Score
    plt.subplot(1, 3, 3)
    plt.plot(r2)
    plt.title(f'{model_name} - R2 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_metrics.png')
    plt.savefig(save_path)
    print(f"Gráfico de métricas salvo em: {save_path}")
    plt.close()
