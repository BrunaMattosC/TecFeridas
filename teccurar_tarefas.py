"""
teccurar_tarefas.py
====================
Catálogo das etapas do projeto e execução delas como subprocesso.

Descreve o que o TecCurar sabe fazer — verificar ambiente, preparar dados,
recriar máscaras, calibrar, validar, treinar, avaliar, inferir em lote — e
sabe executar cada etapa transmitindo a saída em tempo real.

Sem nenhuma dependência de interface gráfica: o painel em
:mod:`teccurar_console` apenas consome o que está aqui. Assim o catálogo
pode ser usado por um script, e a GUI não precisa conhecer linha de comando.

Cada etapa é executada como um **processo separado** rodando o script que o
projeto já tem. É o que evita duplicar lógica: o botão "Treinar" roda o
mesmo ``train.py`` que a linha de comando roda, com os mesmos parâmetros e
os mesmos efeitos. E um treino travado ou cancelado não derruba a janela.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

MOEDAS_DIR = config.BASE_DIR / "Moedas Parâmetro"
MASCARAS_NOVAS_DIR = config.BASE_DIR / "TecCurar_NewMasks"
RESULTADOS_DIR = config.BASE_DIR / "Resultados"
CALIBRACAO_CSV = RESULTADOS_DIR / "Moedas" / "Calibracao" / "calibracao_por_moeda.csv"
RELATORIO_VALIDACAO = (RESULTADOS_DIR / "Relatorios" / "Validacao"
                       / "relatorio_calibracao_comparacao.pdf")


# --------------------------------------------------------------------------- #
# Estado do projeto
# --------------------------------------------------------------------------- #

@dataclass
class Artefato:
    """Um produto do projeto e se ele já existe."""

    chave: str
    titulo: str
    pronto: bool
    detalhe: str


def _contar(pasta: Path, padrao: str = "*") -> int:
    """Conta arquivos reais numa pasta, ignorando marcadores ``.gitkeep``."""
    if not pasta.exists():
        return 0
    return sum(1 for p in pasta.glob(padrao) if p.is_file() and p.name != ".gitkeep")


def checkpoint_mais_recente() -> Optional[Path]:
    """Devolve o checkpoint mais recente, ou ``None`` se não houver nenhum."""
    candidatos = sorted(config.CHECKPOINTS_DIR.glob("*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidatos[0] if candidatos else None


def estado_projeto() -> Dict[str, Artefato]:
    """Fotografa o que já existe em disco.

    É o que permite o painel dizer o que falta fazer em vez de oferecer
    botões que falhariam por falta de pré-requisito.
    """
    n_imagens = _contar(config.IMAGES_DIR)
    n_mascaras = _contar(config.MASKS_DIR)
    n_novas = _contar(MASCARAS_NOVAS_DIR)
    n_moedas = sum(1 for p in MOEDAS_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    ckpt = checkpoint_mais_recente()

    return {a.chave: a for a in [
        Artefato("dataset", "Dataset do Kaggle", n_imagens > 0 and n_mascaras > 0,
                 "%d imagens e %d máscaras" % (n_imagens, n_mascaras) if n_imagens
                 else "não baixado"),
        Artefato("mascaras_novas", "Máscaras recriadas", n_novas > 0,
                 "%d máscaras em TecCurar_NewMasks" % n_novas if n_novas
                 else "ainda não geradas"),
        Artefato("moedas", "Fotografias das moedas", n_moedas > 0,
                 "%d fotografias" % n_moedas if n_moedas else "pasta ausente ou vazia"),
        Artefato("calibracao", "Calibração por moedas", CALIBRACAO_CSV.exists(),
                 "escala estabelecida" if CALIBRACAO_CSV.exists() else "não calculada"),
        Artefato("validacao", "Relatório de validação", RELATORIO_VALIDACAO.exists(),
                 "PDF disponível" if RELATORIO_VALIDACAO.exists() else "não gerado"),
        Artefato("modelo", "Modelo treinado", ckpt is not None,
                 ckpt.name if ckpt else "nenhum checkpoint em 'checkpoints'"),
    ]}


# --------------------------------------------------------------------------- #
# Catálogo de etapas
# --------------------------------------------------------------------------- #

@dataclass
class Etapa:
    """Uma ação do projeto que o painel pode disparar."""

    chave: str
    titulo: str
    descricao: str
    script: str
    argumentos: List[str] = field(default_factory=list)
    requer: List[str] = field(default_factory=list)
    produz: str = ""
    demorada: bool = False
    confirmar: str = ""
    # Constrói argumentos que dependem do estado (ex.: checkpoint atual).
    argumentos_dinamicos: Optional[Callable[[], List[str]]] = None

    def indisponivel_por(self, estado: Dict[str, Artefato]) -> Optional[str]:
        """Explica por que a etapa não pode rodar agora, ou ``None``."""
        faltando = [estado[c].titulo for c in self.requer
                    if c in estado and not estado[c].pronto]
        if not faltando:
            return None
        return "Depende de: %s" % ", ".join(faltando)

    def comando(self, extras: Optional[List[str]] = None) -> List[str]:
        """Monta a linha de comando completa da etapa."""
        args = list(self.argumentos)
        if self.argumentos_dinamicos:
            args += self.argumentos_dinamicos()
        return [sys.executable, str(config.BASE_DIR / self.script)] + args + list(extras or [])


def _args_checkpoint() -> List[str]:
    ckpt = checkpoint_mais_recente()
    return ["--checkpoint", str(ckpt)] if ckpt else []


ETAPAS: List[Etapa] = [
    Etapa(
        "ambiente", "1. Verificar ambiente",
        "Confere dependências, GPU, credenciais do Kaggle e o que já está pronto "
        "nesta máquina. Não altera nada.",
        "verificar_ambiente.py",
    ),
    Etapa(
        "dataset", "2. Baixar e organizar o dataset",
        "Baixa o dataset de feridas do Kaggle, descompacta e separa imagens e "
        "máscaras. Exige credenciais do Kaggle configuradas.",
        "dataset.py", produz="dataset/images e dataset/masks", demorada=True,
    ),
    Etapa(
        "explorar", "3. Explorar os dados",
        "Gera histogramas de área das feridas e um grid de exemplos com a máscara "
        "sobreposta, para inspeção do conjunto.",
        "explore_data.py", requer=["dataset"], produz="outputs/figures",
    ),
    Etapa(
        "mascaras", "4. Recriar as máscaras",
        "Parte das máscaras originais e agrega regiões da lesão que ficaram de "
        "fora, comparando cada pixel com os perfis de ferida, pele e fundo. "
        "As originais não são alteradas.",
        "recriar_mascaras_kaggle.py", argumentos=["--amostras", "12"],
        requer=["dataset"], produz="TecCurar_NewMasks", demorada=True,
    ),
    Etapa(
        "calibrar", "5. Calibrar pelas moedas",
        "Mede o diâmetro das moedas em pixels e confronta com as dimensões "
        "oficiais, estabelecendo a relação pixels/cm.",
        "calibrar_moedas.py", argumentos=["--mascaras-por-moeda", "12"],
        requer=["moedas"], produz="Resultados/Moedas", demorada=True,
    ),
    Etapa(
        "validar", "6. Relatório de validação",
        "Compara máscara original e recriada em 10 imagens, com as sobreposições "
        "sobre a fotografia, medidas em pixels e centímetros, IoU e Dice.",
        "comparar_mascaras_relatorio.py", argumentos=["--n-imagens", "10"],
        requer=["dataset", "mascaras_novas", "calibracao"],
        produz="Resultados/Relatorios/Validacao",
    ),
    Etapa(
        "treinar", "7. Treinar o modelo",
        "Treina a rede de segmentação sobre o dataset e salva o melhor checkpoint. "
        "É a etapa mais longa do projeto.",
        "train.py", requer=["dataset"], produz="checkpoints", demorada=True,
        confirmar="O treinamento pode levar horas e usará a GPU de forma intensiva.\n\n"
                  "Ele roda em segundo plano e você pode interrompê-lo a qualquer "
                  "momento pelo botão Parar — o melhor checkpoint até ali fica salvo.\n\n"
                  "Deseja iniciar agora?",
    ),
    Etapa(
        "avaliar", "8. Avaliar no conjunto de teste",
        "Mede o modelo treinado no conjunto de teste: Dice, IoU, matriz de "
        "confusão, curvas ROC e Precision-Recall, e relatório por imagem.",
        "evaluate.py", requer=["dataset", "modelo"], produz="outputs/reports e figures",
        demorada=True, argumentos_dinamicos=_args_checkpoint,
    ),
    Etapa(
        "lote", "9. Inferência em lote",
        "Segmenta e mede todas as imagens de uma pasta, gerando máscaras, "
        "sobreposições e planilhas de medição. Você escolhe a pasta.",
        "batch_inference.py", requer=["modelo"], produz="outputs/predictions/lote",
        demorada=True, argumentos_dinamicos=_args_checkpoint,
    ),
]

ETAPAS_POR_CHAVE = {e.chave: e for e in ETAPAS}


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #

class ExecutorEtapa:
    """Roda uma etapa como subprocesso e entrega a saída linha a linha.

    A saída chega por uma fila, consumida pela interface no seu próprio
    ritmo. Rodar em processo separado — e não em thread — é o que permite
    interromper um treino de verdade: encerrar uma thread Python no meio de
    um laço de treinamento não é possível de forma limpa.
    """

    def __init__(self) -> None:
        self.fila: "queue.Queue[tuple]" = queue.Queue()
        self._processo: Optional[subprocess.Popen] = None
        self._etapa: Optional[Etapa] = None

    @property
    def rodando(self) -> bool:
        return self._processo is not None and self._processo.poll() is None

    @property
    def etapa_atual(self) -> Optional[Etapa]:
        return self._etapa

    def iniciar(self, etapa: Etapa, extras: Optional[List[str]] = None) -> None:
        """Dispara a etapa. Levanta ``RuntimeError`` se já houver uma rodando."""
        if self.rodando:
            raise RuntimeError("Já existe uma etapa em execução.")

        comando = etapa.comando(extras)
        self._etapa = etapa

        ambiente = os.environ.copy()
        # Sem isto a saída do filho chega só no fim, em blocos, e o painel
        # ficaria parado durante todo um treino.
        ambiente["PYTHONUNBUFFERED"] = "1"
        # Garante UTF-8 na saída: o console do Windows usa cp1252 por padrão
        # e os acentos dos relatórios chegariam corrompidos.
        ambiente["PYTHONIOENCODING"] = "utf-8"

        self._processo = subprocess.Popen(
            comando, cwd=str(config.BASE_DIR), env=ambiente,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.fila.put(("inicio", etapa, " ".join(Path(c).name if i == 1 else c
                                                 for i, c in enumerate(comando))))
        threading.Thread(target=self._bombear, daemon=True).start()

    def _bombear(self) -> None:
        """Lê a saída do processo até o fim e publica na fila."""
        processo, etapa = self._processo, self._etapa
        try:
            for linha in processo.stdout:
                # tqdm reescreve a mesma linha com '\r'; sem isto o painel
                # acumularia centenas de linhas por época.
                texto = linha.rstrip("\n")
                if "\r" in texto:
                    texto = texto.split("\r")[-1]
                if texto.strip():
                    self.fila.put(("saida", etapa, texto))
        except Exception as exc:  # pragma: no cover - falha de leitura do pipe
            self.fila.put(("saida", etapa, "[erro ao ler a saída: %s]" % exc))
        finally:
            codigo = processo.wait()
            self.fila.put(("fim", etapa, codigo))

    def parar(self) -> None:
        """Interrompe a etapa em andamento, se houver."""
        if not self.rodando:
            return
        self.fila.put(("saida", self._etapa, "[interrompido pelo usuário]"))
        self._processo.terminate()
        try:
            self._processo.wait(timeout=8)
        except subprocess.TimeoutExpired:  # pragma: no cover - processo teimoso
            self._processo.kill()
