"""
verificar_ambiente.py
=====================
Diagnostico de ambiente: informa, em qualquer computador, o que ja esta
pronto e o que falta configurar para rodar o projeto.

Executa sem nenhuma dependencia externa -- usa apenas a biblioteca padrao,
de modo que funciona mesmo em um Python recem-instalado, antes de qualquer
`pip install`. As dependencias sao testadas por importacao tolerante.

Uso:
    python verificar_ambiente.py

Codigo de saida:
    0  -> ambiente pronto para pelo menos a medicao
    1  -> falta algo essencial (relatado na saida)

A saida usa apenas caracteres ASCII para nao quebrar em consoles Windows
configurados em cp1252.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata as metadata
import io
import os
import platform
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OK, FALTA, AVISO, INFO = "[ OK   ]", "[FALTA ]", "[AVISO ]", "[INFO  ]"

# Dependencia -> (modulo importavel, versao minima recomendada, grupo)
DEPENDENCIAS = [
    # (pacote no pip,              modulo,                        min,      grupo)
    ("numpy",                      "numpy",                       (1, 26),  "medicao"),
    ("opencv-python-headless",     "cv2",                         (4, 9),   "medicao"),
    ("pandas",                     "pandas",                      (2, 1),   "medicao"),
    ("matplotlib",                 "matplotlib",                  (3, 8),   "graficos"),
    ("scikit-learn",               "sklearn",                     (1, 4),   "graficos"),
    ("torch",                      "torch",                       (2, 1),   "deep learning"),
    ("torchvision",                "torchvision",                 (0, 16),  "deep learning"),
    ("segmentation-models-pytorch", "segmentation_models_pytorch", (0, 3),   "deep learning"),
    ("albumentations",             "albumentations",              (1, 4),   "deep learning"),
    ("torchmetrics",               "torchmetrics",                (1, 3),   "deep learning"),
    ("tensorboard",                "tensorboard",                 (2, 15),  "infra"),
    ("tqdm",                       "tqdm",                        (4, 66),  "infra"),
    ("kaggle",                     "kaggle",                      (2, 0),   "infra"),
    ("pytest",                     "pytest",                      (8, 0),   "testes"),
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Utilitarios
# --------------------------------------------------------------------------- #

def titulo(texto: str) -> None:
    print()
    print("-" * 72)
    print(texto)
    print("-" * 72)


def importar_silencioso(modulo: str):
    """Importa um modulo sem deixar que ele escreva no console.

    Necessario porque alguns pacotes imprimem banners ao serem importados
    -- o cliente ``kaggle``, por exemplo, despeja instrucoes de
    autenticacao inteiras em stdout -- o que destruiria a legibilidade
    deste relatorio.
    """
    saida, erro = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(erro):
        return importlib.import_module(modulo)


def como_tupla(versao: str):
    """Converte '2.13.0+cpu' em (2, 13, 0), ignorando sufixos."""
    numeros = []
    for parte in versao.split("+")[0].split("."):
        digitos = ""
        for ch in parte:
            if ch.isdigit():
                digitos += ch
            else:
                break
        if not digitos:
            break
        numeros.append(int(digitos))
    return tuple(numeros)


# --------------------------------------------------------------------------- #
# Verificacoes
# --------------------------------------------------------------------------- #

def checar_sistema() -> list:
    titulo("1. SISTEMA")
    problemas = []

    v = sys.version_info
    print("%s Python %d.%d.%d (%s)" % (OK if v >= (3, 10) else FALTA,
                                       v.major, v.minor, v.micro, sys.executable))
    if v < (3, 10):
        problemas.append("Python 3.10 ou superior e necessario (o codigo usa 'X | None').")

    print("%s Sistema: %s %s (%s)" % (INFO, platform.system(),
                                      platform.release(), platform.machine()))

    em_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print("%s Ambiente virtual: %s" % (OK if em_venv else AVISO,
                                       "ativo" if em_venv else "NAO detectado"))
    if not em_venv:
        print("         Recomendado: python -m venv .venv  e ativar antes de instalar.")

    try:
        uso = shutil.disk_usage(BASE_DIR)
        livre_gb = uso.free / (1024 ** 3)
        marca = OK if livre_gb >= 5 else (AVISO if livre_gb >= 2 else FALTA)
        print("%s Espaco livre em disco: %.1f GB" % (marca, livre_gb))
        if livre_gb < 5:
            print("         O dataset ocupa ~700 MB e a pilha de deep learning ~1,5 GB.")
    except OSError:
        print("%s Nao foi possivel medir o espaco em disco." % AVISO)

    return problemas


def checar_dependencias() -> tuple:
    titulo("2. DEPENDENCIAS")
    instalados, ausentes = {}, {}

    for pacote, modulo, minimo, grupo in DEPENDENCIAS:
        try:
            importado = importar_silencioso(modulo)
        except Exception:
            ausentes[pacote] = grupo
            print("%s %-30s (grupo: %s)" % (FALTA, pacote, grupo))
            continue

        try:
            versao = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versao = getattr(importado, "__version__", "?")

        atual = como_tupla(versao)
        antigo = bool(atual) and atual < minimo
        marca = AVISO if antigo else OK
        extra = ""
        if antigo:
            extra = "  <- abaixo do minimo %s" % ".".join(map(str, minimo))
        print("%s %-30s %s%s" % (marca, pacote, versao, extra))
        instalados[pacote] = versao

    return instalados, ausentes


def checar_gpu(tem_torch: bool) -> bool:
    """Relata o estado da aceleracao por GPU.

    Delega a deteccao a :mod:`hardware` -- a mesma fonte de verdade usada
    pelo treino --, de modo que este relatorio nunca discorde do que
    ``train.py`` fara de fato. O modulo ``hardware`` so importa a stdlib no
    topo, entao ele pode ser usado aqui mesmo sem o PyTorch instalado.

    Returns:
        ``True`` se ha aceleracao por GPU disponivel para o framework.
    """
    titulo("3. GPU / ACELERACAO")

    try:
        import hardware
    except Exception as exc:  # pragma: no cover - projeto incompleto
        print("%s Nao foi possivel carregar hardware.py: %s" % (AVISO, exc))
        return False

    info = hardware.obter_hardware()

    # 3a. O que a MAQUINA tem -- independente de framework, sem assumir
    #     fabricante nenhum.
    if info.gpus_do_sistema:
        for g in info.gpus_do_sistema:
            memoria = " (~%.1f GB)" % g.memoria_gb if g.memoria_gb else ""
            driver = " driver %s" % g.versao_driver if g.versao_driver else ""
            print("%s Placa no sistema: [%s] %s%s%s"
                  % (INFO, g.fabricante, g.modelo, memoria, driver))
    else:
        print("%s Nenhuma placa de video enumerada pelo sistema operacional." % INFO)

    if info.cuda_do_driver:
        print("%s Driver NVIDIA %s suporta ate CUDA %s"
              % (INFO, info.versao_driver, info.cuda_do_driver))

    if not tem_torch:
        print("%s PyTorch nao instalado -- nao ha como avaliar a aceleracao." % INFO)
        return False

    # 3b. O que o FRAMEWORK consegue usar.
    print("%s Build do PyTorch: %s (CUDA do build: %s)"
          % (INFO, info.versao_framework, info.build_cuda or "NENHUM -- build CPU-only"))

    if not info.aceleracao_disponivel:
        print("%s Aceleracao por GPU INDISPONIVEL. Treino e inferencia usariam CPU." % AVISO)
        print("         Em CPU, o treino completo deste projeto e impraticavel.")
        for problema in info.problemas:
            print("         MOTIVO : %s" % problema)
        for correcao in info.correcoes:
            print("         CORRIJA: %s" % correcao)
        return False

    gpu = info.gpu_principal
    print("%s Backend ativo: %s" % (OK, info.backend.upper()))
    if gpu is not None:
        print("%s GPU utilizavel: %s" % (OK, gpu.nome))
        if gpu.memoria_total_bytes:
            print("%s VRAM: %.2f GB" % (INFO, gpu.memoria_total_gb))
        if gpu.compute_capability:
            print("%s Compute capability: %s" % (INFO, gpu.sm))
        print("%s Mixed precision bf16: %s" % (INFO, "SIM" if gpu.suporta_bf16 else "NAO"))
    if info.versao_cudnn:
        print("%s cuDNN: %s" % (INFO, info.versao_cudnn))

    if info.problemas:
        for problema in info.problemas:
            print("%s %s" % (FALTA, problema))
        for correcao in info.correcoes:
            print("         CORRIJA: %s" % correcao)
        return False

    # 3c. Verificacao REAL: nao basta a API dizer que a GPU existe.
    try:
        device = hardware.resolver_dispositivo(exigir_gpu=False, silencioso=True)
        if hardware.verificar_uso_real_da_gpu(device, verboso=False):
            print("%s Verificacao real: a GPU executou uma operacao de teste." % OK)
        else:
            print("%s A GPU foi detectada mas falhou ao executar a operacao de teste." % FALTA)
            return False
    except Exception as exc:
        print("%s Falha ao exercitar a GPU: %s" % (FALTA, exc))
        return False

    print("%s GPU compativel -- treino acelerado disponivel." % OK)
    print("         Relatorio detalhado: python hardware.py")
    return True


def checar_kaggle() -> None:
    titulo("4. CREDENCIAIS DO KAGGLE")
    achou = False

    if os.environ.get("KAGGLE_API_TOKEN"):
        print("%s KAGGLE_API_TOKEN definido (token moderno, exige kaggle>=2.0)." % OK)
        achou = True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("%s KAGGLE_USERNAME + KAGGLE_KEY definidos." % OK)
        achou = True

    caminho_json = Path.home() / ".kaggle" / "kaggle.json"
    if caminho_json.exists():
        print("%s Arquivo encontrado: %s" % (OK, caminho_json))
        achou = True

    if not achou:
        print("%s Nenhuma credencial configurada." % AVISO)
        print("         So e necessario para BAIXAR o dataset. Configure uma destas:")
        print("           export KAGGLE_API_TOKEN=KGAT_...")
        print("           export KAGGLE_USERNAME=... e export KAGGLE_KEY=...")
        print("           ou salve kaggle.json em ~/.kaggle/kaggle.json")
        print("         Gere em https://www.kaggle.com/settings")


def checar_dados() -> bool:
    titulo("5. DADOS")

    def conta(pasta: Path) -> int:
        if not pasta.exists():
            return 0
        return sum(1 for p in pasta.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    imagens = conta(BASE_DIR / "dataset" / "images")
    mascaras = conta(BASE_DIR / "dataset" / "masks")
    brutos = BASE_DIR / "dataset" / "raw"
    tem_bruto = brutos.exists() and any(brutos.rglob("*.png"))

    print("%s dataset/raw/    : %s" % (OK if tem_bruto else INFO,
                                       "populado" if tem_bruto else "vazio"))
    print("%s dataset/images/ : %d arquivos" % (OK if imagens else FALTA, imagens))
    print("%s dataset/masks/  : %d arquivos" % (OK if mascaras else FALTA, mascaras))

    if not imagens or not mascaras:
        print("         Baixe e organize com: python dataset.py")
        return False

    if imagens != mascaras:
        print("%s Contagens diferentes -- pode haver imagem sem mascara." % AVISO)

    # Pareamento por nome, mesma regra de dataset.list_image_mask_pairs()
    nomes_img = {p.stem.lower() for p in (BASE_DIR / "dataset" / "images").iterdir()
                 if p.suffix.lower() in IMAGE_EXTENSIONS}
    nomes_msk = {p.stem.lower().replace("_mask", "").replace("-mask", "")
                 for p in (BASE_DIR / "dataset" / "masks").iterdir()
                 if p.suffix.lower() in IMAGE_EXTENSIONS}
    pares = len(nomes_img & nomes_msk)
    marca = OK if pares == imagens else AVISO
    print("%s Pareamento por nome: %d pares" % (marca, pares))
    if pares < imagens:
        print("         %d imagens ficariam de fora do treino." % (imagens - pares))
    return pares > 0


def checar_modelo() -> bool:
    titulo("6. MODELO TREINADO")
    pasta = BASE_DIR / "checkpoints"
    ckpts = sorted(pasta.glob("*.pt")) if pasta.exists() else []
    if not ckpts:
        print("%s Nenhum checkpoint (.pt) em checkpoints/." % AVISO)
        print("         Sem ele, predict.py / evaluate.py / batch_inference.py nao rodam.")
        print("         Gere com: python train.py")
        return False
    for c in ckpts:
        print("%s %s (%.1f MB)" % (OK, c.name, c.stat().st_size / 1024 ** 2))
    return True


def veredito(ausentes: dict, tem_dados: bool, tem_modelo: bool, problemas: list,
             tem_gpu: bool = False) -> int:
    titulo("VEREDITO")

    falta_medicao = [p for p, g in ausentes.items() if g == "medicao"]
    falta_dl = [p for p, g in ausentes.items() if g == "deep learning"]
    falta_graficos = [p for p, g in ausentes.items() if g == "graficos"]

    pode_medir = not falta_medicao
    pode_treinar = not (falta_dl or falta_medicao) and tem_dados
    pode_prever = not (falta_dl or falta_medicao) and tem_modelo

    print("%s Medir mascaras (measure.py)      : %s"
          % (OK if pode_medir else FALTA, "SIM" if pode_medir else "NAO"))
    print("%s Rodar a suite de testes          : %s"
          % (OK if pode_medir and "pytest" not in ausentes else FALTA,
             "SIM" if pode_medir and "pytest" not in ausentes else "NAO"))
    print("%s Baixar/explorar o dataset        : %s"
          % (OK if pode_medir else FALTA, "SIM" if pode_medir else "NAO"))
    print("%s Treinar (train.py)               : %s"
          % (OK if pode_treinar else FALTA, "SIM" if pode_treinar else "NAO"))
    print("%s Prever/avaliar em imagens        : %s"
          % (OK if pode_prever else FALTA, "SIM" if pode_prever else "NAO"))
    print("%s Treinar acelerado por GPU        : %s"
          % (OK if tem_gpu else AVISO, "SIM" if tem_gpu else "NAO (CPU)"))

    print()
    if problemas:
        print("ESSENCIAL A RESOLVER:")
        for p in problemas:
            print("  - " + p)
        print()

    passos = []
    if falta_medicao or falta_graficos:
        passos.append("pip install -r requirements.txt")
    elif falta_dl:
        passos.append("pip install -r requirements.txt   (falta a pilha de deep learning)")
    if not tem_dados:
        passos.append("python dataset.py                 (baixa e organiza o dataset)")
    if not tem_modelo:
        passos.append("python train.py                   (gera um checkpoint)")

    if passos:
        print("PROXIMOS PASSOS, NESTA ORDEM:")
        for i, passo in enumerate(passos, 1):
            print("  %d. %s" % (i, passo))
    else:
        print("Ambiente completo: nada pendente.")

    return 1 if (problemas or not pode_medir) else 0


def main() -> int:
    print("=" * 72)
    print("DIAGNOSTICO DE AMBIENTE -- Segmentacao e Medicao de Feridas")
    print("Projeto em: %s" % BASE_DIR)
    print("=" * 72)

    problemas = checar_sistema()
    instalados, ausentes = checar_dependencias()
    tem_gpu = checar_gpu("torch" in instalados)
    checar_kaggle()
    tem_dados = checar_dados()
    tem_modelo = checar_modelo()
    codigo = veredito(ausentes, tem_dados, tem_modelo, problemas, tem_gpu)

    print()
    print("=" * 72)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
