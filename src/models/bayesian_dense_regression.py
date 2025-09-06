import tensorflow as tf
from .bayesian_density_network import BayesianDensityNetwork

class BayesianDenseRegression(tf.keras.Model):
    """
    Um modelo de Regressão Bayesiana Densa (BNN).
    Estima a média (loc) da predição.
    """
    def __init__(self, layer_dims: list, config: dict):
        """
        Args:
            layer_dims (list): Dimensões das camadas densas do núcleo da rede.
            config (dict): Dicionário de configuração do projeto.
        """
        super().__init__()
        self.config = config
        self.loc_net = BayesianDensityNetwork(
            core_layers=layer_dims,
            head_layers=[1],
            config=self.config
        )
        # AJUSTE: Instancia a função de perda como um objeto.
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=None):
        """Executa a passagem forward."""
        return self.loc_net(inputs, training=training)

    def train_step(self, data):
        """Define o passo de treinamento customizado."""
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # AJUSTE: Usa o objeto de perda instanciado.
            rec_loss = self.mse_loss(y, y_pred)
            # A perda KL é a soma de todas as perdas das camadas internas.
            kl_loss = sum(self.loc_net.losses)
            loss = rec_loss + kl_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Atualiza as métricas
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "reconstruction_loss": rec_loss, "kl_loss": kl_loss})
        return results

    def test_step(self, data):
        """Define o passo de validação/teste customizado."""
        x, y = data
        y_pred = self(x, training=False)
        # AJUSTE: Usa o objeto de perda instanciado.
        rec_loss = self.mse_loss(y, y_pred)
        kl_loss = sum(self.loc_net.losses)
        loss = rec_loss + kl_loss

        # Atualiza as métricas
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "reconstruction_loss": rec_loss, "kl_loss": kl_loss})
        return results
