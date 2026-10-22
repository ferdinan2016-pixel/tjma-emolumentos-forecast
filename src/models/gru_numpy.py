"""
gru_numpy.py

Modelo adicional: rede neural recorrente do tipo GRU (Gated Recurrent Unit),
arquitetura mais enxuta que a LSTM (dois portões em vez de três, sem estado
de memória separado do estado oculto), incluída na comparação a pedido do
orientador para ampliar o espectro de arquiteturas recorrentes avaliadas
(Seção 3.3 do Projeto de Qualificação).

Implementação de referência em NumPy puro (forward pass, backpropagation
through time - BPTT - e otimizador Adam implementados manualmente), seguindo
exatamente a mesma convenção da LSTM em lstm_numpy.py, para manter os dois
modelos comparáveis em termos de hiperparâmetros (hidden_size, taxa de
aprendizado, número de épocas) e de custo computacional.

A arquitetura é a de uma GRU de camada única: recebe uma janela de `T`
valores passados (normalizados) e prevê o valor do mês seguinte a partir do
último estado oculto.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GRUForecaster:
    """GRU univariada de camada única para previsão um passo à frente."""

    def __init__(self, hidden_size: int = 16, lr: float = 0.02, seed: int = 42):
        rng = np.random.default_rng(seed)
        H, D = hidden_size, 1  # uma feature de entrada por passo de tempo
        concat = H + D
        escala = 1.0 / np.sqrt(concat)

        self.Wr = rng.uniform(-escala, escala, (H, concat))
        self.Wz = rng.uniform(-escala, escala, (H, concat))
        self.Wh = rng.uniform(-escala, escala, (H, concat))
        self.br = np.zeros((H, 1))
        self.bz = np.zeros((H, 1))
        self.bh = np.zeros((H, 1))
        self.Wy = rng.uniform(-escala, escala, (1, H))
        self.by = np.zeros((1, 1))

        self.H, self.D, self.lr = H, D, lr
        self._params = ["Wr", "Wz", "Wh", "br", "bz", "bh", "Wy", "by"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    def _forward(self, x_seq: np.ndarray):
        T = x_seq.shape[0]
        H = self.H
        h = np.zeros((H, 1))
        cache = []
        for t in range(T):
            xt = np.array([[x_seq[t]]])
            concat1 = np.vstack([h, xt])
            r = _sigmoid(self.Wr @ concat1 + self.br)
            z = _sigmoid(self.Wz @ concat1 + self.bz)
            concat2 = np.vstack([r * h, xt])
            hc = np.tanh(self.Wh @ concat2 + self.bh)
            h_novo = (1 - z) * h + z * hc
            cache.append((concat1, r, z, concat2, hc, h, h_novo))
            h = h_novo
        y = self.Wy @ h + self.by
        return y, h, cache

    def _backward(self, y_true: float, y_pred: np.ndarray, h_final: np.ndarray, cache: list):
        H = self.H
        T = len(cache)
        grads = {p: np.zeros_like(getattr(self, p)) for p in self._params}

        dy = (y_pred - y_true)
        grads["Wy"] = dy @ h_final.T
        grads["by"] = dy
        dh_next = self.Wy.T @ dy

        for t in reversed(range(T)):
            concat1, r, z, concat2, hc, h_prev, h_novo = cache[t]

            dz = dh_next * (hc - h_prev)
            dhc = dh_next * z
            dh_prev_total = dh_next * (1 - z)

            dhc_raw = dhc * (1 - hc ** 2)
            grads["Wh"] += dhc_raw @ concat2.T
            grads["bh"] += dhc_raw
            dconcat2 = self.Wh.T @ dhc_raw
            d_r_h_prev = dconcat2[:H]

            dr = d_r_h_prev * h_prev
            dh_prev_total = dh_prev_total + d_r_h_prev * r

            dr_raw = dr * r * (1 - r)
            grads["Wr"] += dr_raw @ concat1.T
            grads["br"] += dr_raw
            dconcat1_r = self.Wr.T @ dr_raw
            dh_prev_total = dh_prev_total + dconcat1_r[:H]

            dz_raw = dz * z * (1 - z)
            grads["Wz"] += dz_raw @ concat1.T
            grads["bz"] += dz_raw
            dconcat1_z = self.Wz.T @ dz_raw
            dh_prev_total = dh_prev_total + dconcat1_z[:H]

            dh_next = dh_prev_total

        for p in grads:
            np.clip(grads[p], -5, 5, out=grads[p])
        return grads

    def _adam_step(self, grads: dict):
        self._t += 1
        for p in self._params:
            g = grads[p]
            self._m[p] = self._beta1 * self._m[p] + (1 - self._beta1) * g
            self._v[p] = self._beta2 * self._v[p] + (1 - self._beta2) * (g ** 2)
            mhat = self._m[p] / (1 - self._beta1 ** self._t)
            vhat = self._v[p] / (1 - self._beta2 ** self._t)
            atualizado = getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + self._eps)
            setattr(self, p, atualizado)

    def treinar(self, X: np.ndarray, y: np.ndarray, epocas: int = 200, seed: int = 42, verbose: bool = False):
        """Treina por mini-lote de tamanho 1 (janela a janela), com
        embaralhamento por época (o embaralhamento é das AMOSTRAS/janelas de
        treino, não dos pontos dentro de cada janela - a ordem temporal
        interna de cada janela é sempre preservada)."""
        rng = np.random.default_rng(seed)
        n = len(X)
        historico_loss = []
        for epoca in range(epocas):
            ordem = rng.permutation(n)
            perda_epoca = 0.0
            for idx in ordem:
                y_pred, h_final, cache = self._forward(X[idx])
                perda_epoca += float(0.5 * (y_pred[0, 0] - y[idx]) ** 2)
                grads = self._backward(y[idx], y_pred, h_final, cache)
                self._adam_step(grads)
            historico_loss.append(perda_epoca / n)
            if verbose and (epoca % 20 == 0 or epoca == epocas - 1):
                print(f"  época {epoca:4d}  loss médio = {historico_loss[-1]:.6f}")
        return historico_loss

    def prever(self, x_seq: np.ndarray) -> float:
        y_pred, _, _ = self._forward(x_seq)
        return float(y_pred[0, 0])
