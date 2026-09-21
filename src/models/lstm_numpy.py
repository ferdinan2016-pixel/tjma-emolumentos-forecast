"""
lstm_numpy.py

Modelo 2: rede neural recorrente do tipo LSTM (Long Short-Term Memory),
conforme Seção 3.3.2 do Projeto de Qualificação.

Implementação de referência em NumPy puro (forward pass, backpropagation
through time - BPTT - e otimizador Adam implementados manualmente), o que
evita a instalação de frameworks pesados (PyTorch/TensorFlow, que arrastam
dependências de GPU/CUDA de vários gigabytes) neste ambiente de prova de
conceito. Para a fase de dissertação, com a base real e maior volume de
dados, recomenda-se reimplementar em PyTorch ou TensorFlow/Keras, o que
facilita regularização (dropout), empilhamento de camadas e uso de GPU.

A arquitetura é a de uma LSTM de camada única: recebe uma janela de `T`
valores passados (normalizados) e prevê o valor do mês seguinte a partir do
último estado oculto.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMForecaster:
    """LSTM univariada de camada única para previsão um passo à frente."""

    def __init__(self, hidden_size: int = 16, lr: float = 0.01, seed: int = 42):
        rng = np.random.default_rng(seed)
        H, D = hidden_size, 1  # uma feature de entrada por passo de tempo
        concat = H + D
        escala = 1.0 / np.sqrt(concat)

        self.Wf = rng.uniform(-escala, escala, (H, concat))
        self.Wi = rng.uniform(-escala, escala, (H, concat))
        self.Wo = rng.uniform(-escala, escala, (H, concat))
        self.Wg = rng.uniform(-escala, escala, (H, concat))
        self.bf = np.zeros((H, 1))
        self.bi = np.zeros((H, 1))
        self.bo = np.zeros((H, 1))
        self.bg = np.zeros((H, 1))
        self.Wy = rng.uniform(-escala, escala, (1, H))
        self.by = np.zeros((1, 1))

        self.H, self.D, self.lr = H, D, lr
        self._params = ["Wf", "Wi", "Wo", "Wg", "bf", "bi", "bo", "bg", "Wy", "by"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    def _forward(self, x_seq: np.ndarray):
        T = x_seq.shape[0]
        H = self.H
        h = np.zeros((H, 1))
        c = np.zeros((H, 1))
        cache = []
        for t in range(T):
            xt = np.array([[x_seq[t]]])
            z = np.vstack([h, xt])
            f = _sigmoid(self.Wf @ z + self.bf)
            i = _sigmoid(self.Wi @ z + self.bi)
            o = _sigmoid(self.Wo @ z + self.bo)
            g = np.tanh(self.Wg @ z + self.bg)
            c_novo = f * c + i * g
            h_novo = o * np.tanh(c_novo)
            cache.append((z, f, i, o, g, c, c_novo, h_novo))
            h, c = h_novo, c_novo
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
        dc_next = np.zeros((H, 1))

        for t in reversed(range(T)):
            z, f, i, o, g, c_prev, c_novo, h_novo = cache[t]
            tanh_c = np.tanh(c_novo)
            do = dh_next * tanh_c
            do_raw = do * o * (1 - o)
            dc = dh_next * o * (1 - tanh_c ** 2) + dc_next
            df = dc * c_prev
            df_raw = df * f * (1 - f)
            di = dc * g
            di_raw = di * i * (1 - i)
            dg = dc * i
            dg_raw = dg * (1 - g ** 2)

            grads["Wf"] += df_raw @ z.T
            grads["Wi"] += di_raw @ z.T
            grads["Wo"] += do_raw @ z.T
            grads["Wg"] += dg_raw @ z.T
            grads["bf"] += df_raw
            grads["bi"] += di_raw
            grads["bo"] += do_raw
            grads["bg"] += dg_raw

            dz = self.Wf.T @ df_raw + self.Wi.T @ di_raw + self.Wo.T @ do_raw + self.Wg.T @ dg_raw
            dh_next = dz[:H]
            dc_next = dc * f

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
