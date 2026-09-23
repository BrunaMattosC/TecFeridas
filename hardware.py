"""
hardware.py
===========
Configuracao CENTRALIZADA de dispositivo (GPU/CPU) do projeto.

Este modulo e a unica fonte de verdade sobre "onde o treino roda". Todos
os demais modulos (``train``, ``evaluate``, ``predict``,
``batch_inference``) obtem seu ``torch.device`` daqui, de modo que nunca
aconteca de uma etapa usar GPU e outra continuar na CPU.

Principios de projeto
---------------------
1. **Nao assume fabricante.** A deteccao acontece em dois niveis
   independentes: (a) o que o SISTEMA OPERACIONAL enxerga de placas de
   video (NVIDIA, AMD, Intel, Apple) e (b) o que o FRAMEWORK instalado
   consegue de fato usar (CUDA, ROCm, XPU, MPS). O diagnostico compara os
   dois e explica a divergencia quando ela existe.

2. **Sem fallback silencioso.** Se o sistema tem uma GPU mas o PyTorch nao
   consegue usa-la, o modulo emite um aviso explicito dizendo *por que*
   (build CPU-only, arquitetura nao compilada, driver antigo...). Com
   ``REQUIRE_GPU`` ligado, ele interrompe a execucao em vez de cair para a
   CPU.

3. **Verificacao real, nao declarativa.** ``verificar_uso_real_da_gpu()``
   executa uma operacao de verdade no dispositivo e confere que memoria
   foi efetivamente alocada na GPU -- ter ``device = "cuda"`` no codigo
   nao e considerado prova de nada.

Saida em texto
--------------
Todo texto impresso usa apenas caracteres Latin-1, para nao quebrar em
consoles Windows configurados em cp1252.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - apenas para type checking
    import torch


LARGURA = 66
_SEP_FORTE = "=" * LARGURA
_SEP_FRACO = "-" * LARGURA


# --------------------------------------------------------------------------- #
# Excecoes
# --------------------------------------------------------------------------- #

class GPUIndisponivelError(RuntimeError):
    """Erro levantado quando a GPU e exigida (``REQUIRE_GPU``) mas nao esta
    disponivel para o framework."""


# --------------------------------------------------------------------------- #
# Estruturas de dados
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GPUDoSistema:
    """Placa de video enxergada pelo SISTEMA OPERACIONAL.

    Preenchida sem envolver o PyTorch, para que seja possivel detectar o
    caso "existe GPU na maquina, mas o framework nao a usa".
    """

    fabricante: str          # "NVIDIA", "AMD", "Intel", "Apple", "Desconhecido"
    modelo: str
    memoria_bytes: Optional[int] = None
    versao_driver: Optional[str] = None

    @property
    def memoria_gb(self) -> Optional[float]:
        if self.memoria_bytes is None:
            return None
        return self.memoria_bytes / (1024 ** 3)


@dataclass(frozen=True)
class GPUDoFramework:
    """Dispositivo de computo que o PyTorch consegue de fato enderecar."""

    indice: int
    nome: str
    memoria_total_bytes: int
    compute_capability: Optional[Tuple[int, int]] = None
    multiprocessadores: Optional[int] = None
    suporta_bf16: bool = False
    arquitetura_compilada_no_build: bool = True

    @property
    def memoria_total_gb(self) -> float:
        return self.memoria_total_bytes / (1024 ** 3)

    @property
    def sm(self) -> str:
        if self.compute_capability is None:
            return "?"
        return "sm_%d%d" % self.compute_capability


@dataclass
class InfoHardware:
    """Retrato completo do ambiente: SO, GPUs, framework e backend ativo."""

    # Sistema
    sistema: str = ""
    versao_sistema: str = ""
    arquitetura: str = ""
    versao_python: str = ""

    # Framework
    framework: str = "PyTorch"
    versao_framework: str = ""
    build_cuda: Optional[str] = None      # torch.version.cuda  (None em build CPU)
    build_hip: Optional[str] = None       # torch.version.hip   (ROCm/AMD)
    versao_cudnn: Optional[str] = None
    arquiteturas_do_build: List[str] = field(default_factory=list)

    # Driver (via nvidia-smi / rocm-smi)
    versao_driver: Optional[str] = None
    cuda_do_driver: Optional[str] = None

    # Deteccao
    gpus_do_sistema: List[GPUDoSistema] = field(default_factory=list)
    gpus_do_framework: List[GPUDoFramework] = field(default_factory=list)
    backend: str = "cpu"                  # "cuda" | "rocm" | "xpu" | "mps" | "cpu"
    aceleracao_disponivel: bool = False

    # Diagnostico
    problemas: List[str] = field(default_factory=list)
    correcoes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #

    @property
    def gpu_principal(self) -> Optional[GPUDoFramework]:
        return self.gpus_do_framework[0] if self.gpus_do_framework else None

    @property
    def nome_gpu(self) -> str:
        gpu = self.gpu_principal
        if gpu is not None:
            return gpu.nome
        for g in self.gpus_do_sistema:
            if g.fabricante != "Intel":  # prefere a dedicada na descricao
                return g.modelo
        if self.gpus_do_sistema:
            return self.gpus_do_sistema[0].modelo
        return "nenhuma"

    @property
    def vram_gb(self) -> Optional[float]:
        gpu = self.gpu_principal
        if gpu is not None:
            return gpu.memoria_total_gb
        for g in self.gpus_do_sistema:
            if g.memoria_gb:
                return g.memoria_gb
        return None

    @property
    def tem_gpu_fisica(self) -> bool:
        """Existe alguma placa de video dedicada no sistema?

        GPUs integradas Intel/AMD de video sao ignoradas nesta pergunta:
        elas existem em praticamente toda maquina e nao servem para
        treinar com o PyTorch instalado.
        """
        return any(g.fabricante in ("NVIDIA", "AMD", "Apple") for g in self.gpus_do_sistema)


# --------------------------------------------------------------------------- #
# Deteccao no nivel do SISTEMA OPERACIONAL (sem envolver o framework)
# --------------------------------------------------------------------------- #

def _executar(cmd: Sequence[str], timeout: float = 15.0) -> Optional[str]:
    """Executa um comando externo e devolve stdout, ou ``None`` em falha."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _fabricante(texto: str) -> str:
    t = texto.lower()
    if "nvidia" in t:
        return "NVIDIA"
    if "amd" in t or "advanced micro devices" in t or "radeon" in t:
        return "AMD"
    if "intel" in t:
        return "Intel"
    if "apple" in t:
        return "Apple"
    return "Desconhecido"


def _gpus_windows() -> List[GPUDoSistema]:
    """Enumera placas de video no Windows via CIM/WMI (Win32_VideoController).

    ``AdapterRAM`` e um uint32 e satura em 4 GiB, entao a memoria daqui e
    apenas indicativa; a VRAM real vem do ``nvidia-smi`` ou do proprio
    PyTorch.
    """
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps is None:
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { \"$($_.AdapterCompatibility)|$($_.Name)|"
        "$($_.AdapterRAM)|$($_.DriverVersion)\" }"
    )
    # -EncodedCommand (UTF-16LE em base64) evita completamente o inferno de
    # escape de aspas entre subprocess, cmd.exe e PowerShell -- o script tem
    # aspas duplas e "$(...)" dentro.
    import base64

    codificado = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    saida = _executar(
        [ps, "-NoProfile", "-NonInteractive", "-EncodedCommand", codificado], timeout=30
    )
    if not saida:
        return []

    gpus: List[GPUDoSistema] = []
    for linha in saida.splitlines():
        partes = linha.strip().split("|")
        if len(partes) < 4 or not partes[1]:
            continue
        fabricante, modelo, ram, driver = partes[0], partes[1], partes[2], partes[3]
        try:
            memoria = int(ram) if ram.strip() else None
        except ValueError:
            memoria = None
        gpus.append(
            GPUDoSistema(
                fabricante=_fabricante(fabricante or modelo),
                modelo=modelo.strip(),
                memoria_bytes=memoria,
                versao_driver=driver.strip() or None,
            )
        )
    return gpus


def _gpus_linux() -> List[GPUDoSistema]:
    """Enumera placas de video no Linux via ``lspci``."""
    saida = _executar(["lspci"])
    if not saida:
        return []
    gpus: List[GPUDoSistema] = []
    for linha in saida.splitlines():
        baixo = linha.lower()
        if not any(k in baixo for k in ("vga compatible controller", "3d controller", "display controller")):
            continue
        descricao = linha.split(":", 2)[-1].strip()
        gpus.append(GPUDoSistema(fabricante=_fabricante(descricao), modelo=descricao))
    return gpus


def _gpus_macos() -> List[GPUDoSistema]:
    """Enumera placas de video no macOS via ``system_profiler``."""
    saida = _executar(["system_profiler", "SPDisplaysDataType"], timeout=30)
    if not saida:
        return []
    gpus: List[GPUDoSistema] = []
    modelo = None
    for linha in saida.splitlines():
        limpa = linha.strip()
        if limpa.startswith("Chipset Model:"):
            modelo = limpa.split(":", 1)[1].strip()
            gpus.append(GPUDoSistema(fabricante=_fabricante(modelo), modelo=modelo))
    return gpus


def detectar_gpus_do_sistema() -> List[GPUDoSistema]:
    """Enumera as placas de video instaladas, seja qual for o fabricante."""
    sistema = platform.system()
    try:
        if sistema == "Windows":
            return _gpus_windows()
        if sistema == "Linux":
            return _gpus_linux()
        if sistema == "Darwin":
            return _gpus_macos()
    except Exception:  # pragma: no cover - deteccao nunca deve derrubar o programa
        return []
    return []


def detectar_driver_nvidia() -> Tuple[Optional[str], Optional[str]]:
    """Le driver e versao de CUDA suportada pelo driver via ``nvidia-smi``.

    Returns:
        Tupla ``(versao_do_driver, versao_cuda_do_driver)``; ``(None, None)``
        se ``nvidia-smi`` nao existir (maquina sem GPU NVIDIA).
    """
    saida = _executar(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    driver = saida.strip().splitlines()[0].strip() if saida and saida.strip() else None

    cuda = None
    cabecalho = _executar(["nvidia-smi"])
    if cabecalho:
        for linha in cabecalho.splitlines():
            if "CUDA Version" in linha:
                cuda = linha.split("CUDA Version:")[-1].replace("|", "").strip()
                break
    return driver, cuda


# --------------------------------------------------------------------------- #
# Deteccao no nivel do FRAMEWORK
# --------------------------------------------------------------------------- #

def _backend_ativo(torch_mod) -> str:
    """Escolhe o backend acelerado disponivel, sem assumir fabricante."""
    if torch_mod.cuda.is_available():
        # torch.version.hip preenchido => build ROCm (GPU AMD) usando a mesma API
        return "rocm" if getattr(torch_mod.version, "hip", None) else "cuda"
    xpu = getattr(torch_mod, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"          # Intel Arc / Data Center GPU
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"          # Apple Silicon
    return "cpu"


def _coletar_gpus_do_framework(torch_mod, backend: str) -> List[GPUDoFramework]:
    gpus: List[GPUDoFramework] = []
    if backend in ("cuda", "rocm"):
        try:
            compiladas = {a.split("_")[-1] for a in torch_mod.cuda.get_arch_list() if a.startswith("sm_")}
        except Exception:
            compiladas = set()

        for i in range(torch_mod.cuda.device_count()):
            props = torch_mod.cuda.get_device_properties(i)
            cc = (props.major, props.minor)
            try:
                bf16 = bool(torch_mod.cuda.is_bf16_supported())
            except Exception:
                bf16 = cc >= (8, 0)
            gpus.append(
                GPUDoFramework(
                    indice=i,
                    nome=props.name,
                    memoria_total_bytes=props.total_memory,
                    compute_capability=cc,
                    multiprocessadores=getattr(props, "multi_processor_count", None),
                    suporta_bf16=bf16,
                    # Em ROCm nao ha lista sm_*; nao ha o que comparar.
                    arquitetura_compilada_no_build=(
                        True if (backend == "rocm" or not compiladas)
                        else ("%d%d" % cc) in compiladas
                    ),
                )
            )
    elif backend == "xpu":
        xpu = torch_mod.xpu
        for i in range(xpu.device_count()):
            props = xpu.get_device_properties(i)
            gpus.append(
                GPUDoFramework(
                    indice=i,
                    nome=getattr(props, "name", "Intel XPU %d" % i),
                    memoria_total_bytes=getattr(props, "total_memory", 0),
                    suporta_bf16=True,
                )
            )
    elif backend == "mps":
        gpus.append(
            GPUDoFramework(
                indice=0,
                nome="Apple Silicon (Metal Performance Shaders)",
                memoria_total_bytes=0,   # MPS usa memoria unificada
                suporta_bf16=False,
            )
        )
    return gpus


# --------------------------------------------------------------------------- #
# Consolidacao + diagnostico
# --------------------------------------------------------------------------- #

def detectar_hardware() -> InfoHardware:
    """Levanta todo o quadro de hardware/software relevante ao treino.

    Nunca levanta excecao: qualquer falha de deteccao vira um item em
    ``problemas``, para que o diagnostico continue util em ambientes
    quebrados.
    """
    info = InfoHardware(
        sistema=platform.system(),
        versao_sistema=platform.release(),
        arquitetura=platform.machine(),
        versao_python=platform.python_version(),
    )

    info.gpus_do_sistema = detectar_gpus_do_sistema()
    if any(g.fabricante == "NVIDIA" for g in info.gpus_do_sistema) or shutil.which("nvidia-smi"):
        info.versao_driver, info.cuda_do_driver = detectar_driver_nvidia()

    try:
        import torch
    except ImportError:
        info.versao_framework = "nao instalado"
        info.problemas.append("PyTorch nao esta instalado.")
        info.correcoes.append("pip install -r requirements.txt")
        return info

    info.versao_framework = torch.__version__
    info.build_cuda = torch.version.cuda
    info.build_hip = getattr(torch.version, "hip", None)
    try:
        info.arquiteturas_do_build = list(torch.cuda.get_arch_list())
    except Exception:
        info.arquiteturas_do_build = []

    info.backend = _backend_ativo(torch)
    info.aceleracao_disponivel = info.backend != "cpu"
    info.gpus_do_framework = _coletar_gpus_do_framework(torch, info.backend)

    if info.backend in ("cuda", "rocm"):
        try:
            cudnn = torch.backends.cudnn.version()
            if cudnn:
                # 91002 -> "9.10.2"
                info.versao_cudnn = "%d.%d.%d" % (cudnn // 10000, (cudnn % 10000) // 100, cudnn % 100)
        except Exception:
            info.versao_cudnn = None

    _diagnosticar(info)
    return info


def _diagnosticar(info: InfoHardware) -> None:
    """Compara "o que a maquina tem" com "o que o framework usa" e explica
    qualquer divergencia com uma correcao acionavel."""
    if info.aceleracao_disponivel:
        gpu = info.gpu_principal
        if gpu is not None and not gpu.arquitetura_compilada_no_build:
            info.problemas.append(
                "A GPU %s (%s) nao esta entre as arquiteturas compiladas neste "
                "build do PyTorch (%s)." % (gpu.nome, gpu.sm, ", ".join(info.arquiteturas_do_build))
            )
            info.correcoes.append(
                "Instale um build do PyTorch que inclua %s." % gpu.sm
            )
        return

    # Sem aceleracao: por que?
    if not info.tem_gpu_fisica:
        info.problemas.append(
            "Nenhuma GPU dedicada foi encontrada neste computador."
        )
        info.correcoes.append(
            "Sem GPU dedicada nao ha o que configurar: o treino rodara em CPU."
        )
        return

    dedicadas = [g for g in info.gpus_do_sistema if g.fabricante in ("NVIDIA", "AMD", "Apple")]
    nomes = ", ".join(g.modelo for g in dedicadas)

    if info.build_cuda is None and info.build_hip is None:
        info.problemas.append(
            "GPU presente (%s), mas o PyTorch instalado (%s) e uma build "
            "SOMENTE CPU -- ele foi compilado sem suporte a GPU."
            % (nomes, info.versao_framework)
        )
        alvo = _indice_pytorch_sugerido(info)
        info.correcoes.append(
            "Reinstale o PyTorch com suporte a GPU:\n"
            "           pip install --index-url %s \\\n"
            "               torch==%s torchvision"
            % (alvo, info.versao_framework.split("+")[0])
        )
        return

    if info.build_cuda is not None:
        info.problemas.append(
            "GPU presente (%s) e PyTorch compilado com CUDA %s, mas "
            "torch.cuda.is_available() retorna False."
            % (nomes, info.build_cuda)
        )
        # Causa frequente e facil de passar despercebida: a variavel esconde
        # as placas do processo, e o resto do ambiente parece perfeito.
        mascara = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mascara is not None and mascara.strip() in ("", "-1"):
            info.correcoes.append(
                "CUDA_VISIBLE_DEVICES esta definida como '%s', o que esconde "
                "todas as GPUs deste processo. Remova a variavel (ou defina-a "
                "como '0') e execute de novo." % mascara
            )
        elif info.versao_driver is None:
            info.correcoes.append(
                "O 'nvidia-smi' nao respondeu: o driver da GPU provavelmente "
                "nao esta instalado ou esta corrompido. Reinstale o driver."
            )
        else:
            info.correcoes.append(
                "Driver %s (CUDA %s) esta presente. Verifique se o CUDA do "
                "build (%s) nao e mais novo que o do driver (%s)."
                % (info.versao_driver, info.cuda_do_driver,
                   info.build_cuda, info.cuda_do_driver)
            )


def _indice_pytorch_sugerido(info: InfoHardware) -> str:
    """Sugere o indice de wheels correto para o driver detectado.

    Um wheel CUDA 13.x exige driver >= 580; com um driver 12.x instalado,
    recomendar cu13x levaria o usuario direto a um segundo erro.
    """
    if any(g.fabricante == "AMD" for g in info.gpus_do_sistema):
        return "https://download.pytorch.org/whl/rocm6.2"
    cuda_driver = info.cuda_do_driver or ""
    try:
        maior = int(cuda_driver.split(".")[0])
    except (ValueError, IndexError):
        maior = 12
    if maior >= 13:
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu126"


# --------------------------------------------------------------------------- #
# Cache do diagnostico (a deteccao envolve subprocessos; nao repetir a toa)
# --------------------------------------------------------------------------- #

_CACHE: Optional[InfoHardware] = None


def obter_hardware(recarregar: bool = False) -> InfoHardware:
    """Devolve o :class:`InfoHardware` do processo, detectando-o na 1a chamada."""
    global _CACHE
    if _CACHE is None or recarregar:
        _CACHE = detectar_hardware()
    return _CACHE


# --------------------------------------------------------------------------- #
# Selecao de dispositivo
# --------------------------------------------------------------------------- #

def resolver_dispositivo(
    preferido: Optional[str] = None,
    exigir_gpu: Optional[bool] = None,
    silencioso: bool = False,
) -> "torch.device":
    """Decide o dispositivo de execucao -- ponto unico de decisao do projeto.

    Args:
        preferido: ``"cuda"``/``"gpu"``, ``"cpu"`` ou ``"auto"``. Se ``None``,
            usa ``config.DEVICE_CFG.preferred``.
        exigir_gpu: se ``True``, levanta :class:`GPUIndisponivelError` em vez
            de cair para a CPU. Se ``None``, usa ``config.DEVICE_CFG.require_gpu``.
        silencioso: suprime o aviso impresso quando ha GPU no sistema mas ela
            nao pode ser usada.

    Returns:
        :class:`torch.device` a ser usado por modelo, dados e metricas.

    Raises:
        GPUIndisponivelError: quando ``exigir_gpu`` e a GPU nao esta utilizavel.
    """
    import torch

    import config

    if preferido is None:
        preferido = config.DEVICE_CFG.preferred
    if exigir_gpu is None:
        exigir_gpu = config.DEVICE_CFG.require_gpu

    info = obter_hardware()
    preferido = (preferido or "auto").lower()

    if preferido == "cpu":
        return torch.device("cpu")

    if info.aceleracao_disponivel:
        if info.backend in ("cuda", "rocm"):
            return torch.device("cuda", config.DEVICE_CFG.gpu_index)
        return torch.device(info.backend)

    # ---- Nao ha aceleracao: nunca cair para CPU em silencio ---- #
    if info.tem_gpu_fisica:
        if not silencioso:
            _avisar_gpu_ociosa(info)
        if exigir_gpu:
            raise GPUIndisponivelError(
                "REQUIRE_GPU esta ativo, mas a GPU detectada nao esta disponivel "
                "para o PyTorch.\n" + _texto_problemas(info) +
                "\nPara treinar em CPU mesmo assim, defina REQUIRE_GPU=0 "
                "(variavel de ambiente) ou require_gpu=False em config.DEVICE_CFG."
            )
    elif exigir_gpu:
        raise GPUIndisponivelError(
            "REQUIRE_GPU esta ativo, mas nenhuma GPU foi encontrada neste computador."
        )

    return torch.device("cpu")


def _texto_problemas(info: InfoHardware) -> str:
    linhas = []
    for problema in info.problemas:
        linhas.append("  MOTIVO : " + problema)
    for correcao in info.correcoes:
        linhas.append("  CORRIJA: " + correcao)
    return "\n".join(linhas)


def _avisar_gpu_ociosa(info: InfoHardware) -> None:
    """Aviso alto e claro do item 9 do escopo: GPU existe e nao esta em uso."""
    print()
    print(_SEP_FORTE)
    print("ATENCAO")
    print("A GPU foi detectada, porem NAO esta sendo utilizada pelo treinamento.")
    print("Verifique a instalacao/configuracao do ambiente.")
    print(_SEP_FRACO)
    print(_texto_problemas(info))
    print(_SEP_FORTE)
    print()


# --------------------------------------------------------------------------- #
# Banner de configuracao de hardware
# --------------------------------------------------------------------------- #

def _rotulo_dispositivo(device: "torch.device") -> str:
    return "CPU" if device.type == "cpu" else "GPU"


def banner_hardware(device: "torch.device", info: Optional[InfoHardware] = None) -> str:
    """Monta o bloco 'CONFIGURACAO DE HARDWARE' descrito no escopo."""
    info = info or obter_hardware()
    linhas = [_SEP_FORTE, "CONFIGURACAO DE HARDWARE", _SEP_FORTE, ""]

    usando_gpu = device.type != "cpu"
    linhas.append("GPU disponivel: %s" % ("SIM" if info.aceleracao_disponivel else "NAO"))

    if info.aceleracao_disponivel:
        gpu = info.gpu_principal
        linhas.append("GPU: %s" % (gpu.nome if gpu else info.nome_gpu))
        if gpu and gpu.memoria_total_bytes:
            linhas.append("Memoria: %.2f GB" % gpu.memoria_total_gb)
        if gpu and gpu.compute_capability:
            linhas.append("Compute capability: %s" % gpu.sm)
        if info.build_cuda:
            linhas.append("CUDA: %s (driver: %s)" % (info.build_cuda, info.cuda_do_driver or "?"))
        if info.build_hip:
            linhas.append("ROCm/HIP: %s" % info.build_hip)
        if info.versao_cudnn:
            linhas.append("cuDNN: %s" % info.versao_cudnn)
        if info.versao_driver:
            linhas.append("Driver: %s" % info.versao_driver)
    elif info.gpus_do_sistema:
        linhas.append("GPU fisica no sistema: %s" % info.nome_gpu)

    linhas.append("Framework: %s %s" % (info.framework, info.versao_framework))
    linhas.append("")
    linhas.append("Dispositivo selecionado: %s" % _rotulo_dispositivo(device))

    if not usando_gpu:
        linhas.append("")
        linhas.append("ATENCAO: o treinamento sera executado utilizando CPU.")
        if info.tem_gpu_fisica:
            linhas.append(_texto_problemas(info))

    linhas.append("")
    linhas.append(_SEP_FORTE)
    return "\n".join(linhas)


def imprimir_banner_hardware(device: "torch.device", info: Optional[InfoHardware] = None) -> None:
    print(banner_hardware(device, info))


# --------------------------------------------------------------------------- #
# Ajuste dos backends (TF32, cuDNN)
# --------------------------------------------------------------------------- #

def ajustar_backends(device: "torch.device") -> None:
    """Aplica os ajustes de backend definidos em ``config.DEVICE_CFG``.

    Nenhum destes ajustes altera a arquitetura, a loss ou as metricas do
    projeto -- eles so mudam *como* as mesmas contas sao executadas.
    """
    import torch

    import config

    if device.type != "cuda":
        return

    cfg = config.DEVICE_CFG

    # TF32: multiplicacoes de matriz/convolucao em precisao reduzida no
    # acumulador FP32 dos Tensor Cores (Ampere+). Ganho grande, impacto
    # numerico muito menor que FP16.
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark testa algoritmos de convolucao e fixa o mais rapido.
    # Exige shapes constantes (o caso deste projeto: 512x512 fixo) e e
    # INCOMPATIVEL com determinismo -- por isso vem desligado por padrao,
    # preservando a reprodutibilidade original do projeto.
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# --------------------------------------------------------------------------- #
# Mixed precision
# --------------------------------------------------------------------------- #

def configuracao_amp(device: "torch.device") -> Tuple[bool, Optional["torch.dtype"], str]:
    """Decide se o treino usa precisao mista e em qual dtype.

    Regras:
        - AMP so e habilitado em dispositivo acelerado (nunca em CPU, onde
          traria lentidao em vez de ganho);
        - ``fp16`` exige Tensor Cores (compute capability >= 7.0);
        - ``bf16`` exige compute capability >= 8.0 (Ampere+);
        - ``auto`` escolhe ``bf16`` quando disponivel apenas se pedido
          explicitamente em ``config``; o padrao do projeto e ``fp16``,
          que e o que o codigo original ja pressupunha (GradScaler).

    Returns:
        Tupla ``(habilitado, dtype, motivo)``.
    """
    import torch

    import config

    cfg = config.DEVICE_CFG
    if not config.TRAIN_CFG.use_amp:
        return False, None, "desabilitado em TRAIN_CFG.use_amp"
    if device.type != "cuda":
        return False, None, "AMP so e usado em GPU (dispositivo atual: %s)" % device.type

    info = obter_hardware()
    gpu = info.gpu_principal
    cc = gpu.compute_capability if gpu else None
    bf16_ok = bool(gpu and gpu.suporta_bf16)

    # Antes de Volta (sm_70) nao ha Tensor Cores: AMP nao acelera nada e so
    # adiciona risco numerico. Melhor treinar em fp32.
    if cc is not None and cc < (7, 0):
        return False, None, "GPU sem Tensor Cores (sm_%d%d); AMP desligado" % cc

    escolhido = (cfg.amp_dtype or "fp16").lower()
    if escolhido == "auto":
        escolhido = "bf16" if bf16_ok else "fp16"

    if escolhido == "bf16":
        if not bf16_ok:
            return True, torch.float16, "bf16 nao suportado por esta GPU; usando fp16"
        return True, torch.bfloat16, "bf16 (compute capability %s), sem GradScaler" % (
            "sm_%d%d" % cc if cc else "?"
        )

    return True, torch.float16, "fp16 com GradScaler"


def contexto_autocast(device: "torch.device", habilitado: bool, dtype: Optional["torch.dtype"] = None):
    """Contexto de autocast compativel com PyTorch antigo e novo.

    A partir do 2.4, ``torch.cuda.amp.autocast`` esta depreciado em favor
    de ``torch.amp.autocast("cuda", ...)``.
    """
    import torch

    tipo = device.type if device.type != "cpu" else "cpu"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        if dtype is not None:
            return torch.amp.autocast(tipo, enabled=habilitado, dtype=dtype)
        return torch.amp.autocast(tipo, enabled=habilitado)
    return torch.cuda.amp.autocast(enabled=habilitado)  # pragma: no cover - torch < 2.4


def criar_grad_scaler(device: "torch.device", habilitado: bool, dtype: Optional["torch.dtype"] = None):
    """Cria o ``GradScaler``. Com bf16 o scaler e desnecessario (mesmo
    expoente do fp32) e fica desabilitado."""
    import torch

    ativo = bool(habilitado and device.type == "cuda" and dtype is not torch.bfloat16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=ativo)
    return torch.cuda.amp.GradScaler(enabled=ativo)  # pragma: no cover - torch < 2.4


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #

def parametros_dataloader(device: "torch.device", treino: bool = True) -> Dict[str, Any]:
    """Parametros de ``DataLoader`` adequados ao dispositivo em uso.

    Em GPU vale a pena manter workers vivos e pinar memoria (habilita a
    copia assincrona host->device); em CPU esses recursos so adicionam
    custo. Os valores vem de ``config.DATA_CFG``, nao sao inventados aqui.
    """
    import config

    cfg = config.DATA_CFG
    usando_gpu = device.type != "cpu"

    num_workers = cfg.num_workers if usando_gpu else min(cfg.num_workers, 2)
    params: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": num_workers,
        # pin_memory so tem efeito (e so faz sentido) com GPU de destino.
        "pin_memory": bool(cfg.pin_memory and usando_gpu),
        # Descartar o ultimo lote incompleto evita recompilar kernels cuDNN
        # para um tamanho diferente; so no treino, para nao perder amostras
        # de validacao/teste.
        "drop_last": bool(treino and usando_gpu and cfg.drop_last_train),
    }
    if num_workers > 0:
        params["persistent_workers"] = bool(cfg.persistent_workers)
        params["prefetch_factor"] = int(cfg.prefetch_factor)
    return params


def para_dispositivo(tensor, device: "torch.device", nao_bloqueante: Optional[bool] = None):
    """Move um tensor para o dispositivo, de forma assincrona quando possivel.

    ``non_blocking=True`` so tem efeito real se o tensor de origem estiver
    em memoria "pinada" -- o que ``pin_memory=True`` no DataLoader garante.
    """
    import config

    if nao_bloqueante is None:
        nao_bloqueante = device.type == "cuda" and config.DATA_CFG.pin_memory
    return tensor.to(device, non_blocking=nao_bloqueante)


# --------------------------------------------------------------------------- #
# Memoria e utilizacao da GPU
# --------------------------------------------------------------------------- #

def memoria_gpu(device: "torch.device") -> Dict[str, float]:
    """Fotografia da memoria da GPU, em MB.

    Returns:
        Dicionario com ``alocada``, ``reservada``, ``pico_alocado``,
        ``total`` e ``livre``. Vazio se o dispositivo nao for CUDA.
    """
    import torch

    if device.type != "cuda":
        return {}
    mb = 1024 ** 2
    livre, total = torch.cuda.mem_get_info(device)
    return {
        "alocada": torch.cuda.memory_allocated(device) / mb,
        "reservada": torch.cuda.memory_reserved(device) / mb,
        "pico_alocado": torch.cuda.max_memory_allocated(device) / mb,
        "pico_reservado": torch.cuda.max_memory_reserved(device) / mb,
        "total": total / mb,
        "livre": livre / mb,
    }


def texto_memoria_gpu(device: "torch.device") -> str:
    """Resumo de uma linha do uso de memoria, pronto para log."""
    m = memoria_gpu(device)
    if not m:
        return "n/a (CPU)"
    return (
        "alocada %.0f MB | reservada %.0f MB | pico %.0f MB | livre %.0f/%.0f MB"
        % (m["alocada"], m["reservada"], m["pico_alocado"], m["livre"], m["total"])
    )


def _utilizacao_via_nvml(device: "torch.device") -> Optional[float]:
    """Ocupacao via NVML -- barata (~microssegundos), boa para amostrar."""
    import torch

    try:
        return float(torch.cuda.utilization(device))
    except Exception:
        return None


def _utilizacao_via_smi(device: "torch.device") -> Optional[float]:
    """Ocupacao via ``nvidia-smi`` -- cria um processo, use com parcimonia."""
    saida = _executar(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        timeout=5,
    )
    if not saida:
        return None
    try:
        return float(saida.strip().splitlines()[device.index or 0])
    except (ValueError, IndexError):
        return None


def utilizacao_gpu(device: "torch.device") -> Optional[float]:
    """Percentual de ocupacao do nucleo grafico, ou ``None`` se indisponivel.

    Tenta primeiro a NVML e cai para ``nvidia-smi`` se a NVML nao estiver
    acessivel.
    """
    if device.type != "cuda":
        return None
    valor = _utilizacao_via_nvml(device)
    return valor if valor is not None else _utilizacao_via_smi(device)


class AmostradorGPU:
    """Amostra a ocupacao da GPU em uma thread, durante uma epoca.

    A media de ocupacao coletada enquanto o treino roda e a evidencia mais
    direta de que a GPU esta realmente trabalhando -- diferente de apenas
    imprimir ``device = cuda``.

    Uso::

        with AmostradorGPU(device) as amostrador:
            ...treina uma epoca...
        print(amostrador.media, amostrador.maximo)
    """

    def __init__(self, device: "torch.device", intervalo: Optional[float] = None) -> None:
        self.device = device
        self.intervalo = intervalo
        self.amostras: List[float] = []
        self._medir = None
        self._parar = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def ativo(self) -> bool:
        return self._thread is not None

    def _laco(self) -> None:
        while not self._parar.wait(self.intervalo):
            valor = self._medir(self.device)
            if valor is not None:
                self.amostras.append(valor)

    def __enter__(self) -> "AmostradorGPU":
        if self.device.type != "cuda":
            return self

        # A fonte de medicao e escolhida UMA vez: a NVML custa
        # microssegundos e pode ser amostrada a cada 0,5 s; o nvidia-smi
        # cria um processo por leitura, entao e amostrado bem mais devagar
        # para nao competir com o proprio treino pela CPU.
        if _utilizacao_via_nvml(self.device) is not None:
            self._medir, padrao = _utilizacao_via_nvml, 0.5
        elif _utilizacao_via_smi(self.device) is not None:
            self._medir, padrao = _utilizacao_via_smi, 3.0
        else:
            return self

        if self.intervalo is None:
            self.intervalo = padrao
        self._thread = threading.Thread(target=self._laco, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    @property
    def media(self) -> Optional[float]:
        return sum(self.amostras) / len(self.amostras) if self.amostras else None

    @property
    def maximo(self) -> Optional[float]:
        return max(self.amostras) if self.amostras else None

    def resumo(self) -> str:
        if not self.amostras:
            return "n/a"
        return "media %.0f%% | pico %.0f%%" % (self.media, self.maximo)


# --------------------------------------------------------------------------- #
# Verificacao REAL de uso da GPU (item 8 do escopo)
# --------------------------------------------------------------------------- #

def verificar_uso_real_da_gpu(device: "torch.device", verboso: bool = True) -> bool:
    """Prova, executando trabalho de verdade, que o dispositivo esta ativo.

    Nao confia em ``device.type == "cuda"``: aloca tensores, executa uma
    multiplicacao de matrizes, sincroniza e confere que (a) o resultado
    reside na GPU e (b) a memoria alocada na GPU cresceu.

    Returns:
        ``True`` se a GPU executou a operacao; ``False`` em CPU ou falha.
    """
    import torch

    if device.type != "cuda":
        if verboso:
            print("[VERIFICACAO] Dispositivo em uso: CPU -- nada a verificar na GPU.")
        return False

    try:
        antes = torch.cuda.memory_allocated(device)
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)

        # Aquecimento: a PRIMEIRA operacao em uma GPU paga a criacao do
        # contexto CUDA e o carregamento dos kernels (centenas de ms).
        # Cronometrar so ela reportaria um desempenho dezenas de vezes
        # abaixo do real.
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize(device)

        repeticoes = 10
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            c = a @ b
        torch.cuda.synchronize(device)
        decorrido = (time.perf_counter() - inicio) / repeticoes

        depois = torch.cuda.memory_allocated(device)
        ok = bool(c.is_cuda) and depois > antes and torch.isfinite(c).all().item()

        # ~2 * N^3 FLOPs na multiplicacao de duas matrizes N x N
        tflops = (2 * 2048 ** 3) / decorrido / 1e12 if decorrido > 0 else float("nan")

        del a, b, c
        torch.cuda.empty_cache()

        if verboso:
            print("[VERIFICACAO] Operacao executada em %s" % torch.cuda.get_device_name(device))
            print("[VERIFICACAO] Memoria da GPU alocada durante o teste: %.1f MB"
                  % ((depois - antes) / 1024 ** 2))
            print("[VERIFICACAO] matmul 2048x2048: %.1f ms (%.2f TFLOP/s)"
                  % (decorrido * 1000, tflops))
            print("[VERIFICACAO] GPU realmente em uso: %s" % ("SIM" if ok else "NAO"))
        return ok
    except Exception as exc:
        if verboso:
            print("[VERIFICACAO] Falha ao exercitar a GPU: %s" % exc)
        return False


def conferir_modelo_no_dispositivo(model, device: "torch.device", nome: str = "modelo") -> None:
    """Confere que TODOS os parametros e buffers estao no dispositivo certo.

    Protege contra o cenario descrito no item 5 do escopo (modelo em um
    dispositivo e dados em outro), que em geral so aparece como um
    ``RuntimeError`` obscuro no meio da primeira epoca.

    Raises:
        RuntimeError: se algum parametro/buffer estiver fora de ``device``.
    """
    errados = []
    for nome_param, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != device.type:
            errados.append("%s (%s)" % (nome_param, tensor.device))
    if errados:
        raise RuntimeError(
            "%s nao esta inteiramente em %s. Tensores fora do dispositivo: %s"
            % (nome, device, ", ".join(errados[:5]))
        )


def conferir_lote_no_dispositivo(device: "torch.device", **tensores) -> None:
    """Confere que os tensores de um lote estao no dispositivo esperado.

    Raises:
        RuntimeError: descrevendo exatamente qual tensor ficou para tras.
    """
    errados = [
        "%s (%s)" % (nome, t.device)
        for nome, t in tensores.items()
        if t is not None and hasattr(t, "device") and t.device.type != device.type
    ]
    if errados:
        raise RuntimeError(
            "Dados fora do dispositivo do modelo (%s): %s. "
            "Modelo e dados precisam estar no MESMO dispositivo."
            % (device, ", ".join(errados))
        )


# --------------------------------------------------------------------------- #
# Relatorio de diagnostico (item 12 do escopo)
# --------------------------------------------------------------------------- #

def relatorio_diagnostico(info: Optional[InfoHardware] = None) -> str:
    """Monta o bloco 'DIAGNOSTICO DO AMBIENTE' completo, em texto."""
    import config

    info = info or obter_hardware()
    linhas = [_SEP_FORTE, "DIAGNOSTICO DO AMBIENTE", _SEP_FORTE, ""]

    linhas.append("Python: %s" % info.versao_python)
    linhas.append("Sistema: %s %s (%s)" % (info.sistema, info.versao_sistema, info.arquitetura))
    linhas.append("Framework: %s %s" % (info.framework, info.versao_framework))
    linhas.append("")

    if info.gpus_do_sistema:
        linhas.append("GPUs instaladas no sistema:")
        for g in info.gpus_do_sistema:
            memoria = " (%.1f GB reportados pelo SO)" % g.memoria_gb if g.memoria_gb else ""
            linhas.append("  - [%s] %s%s" % (g.fabricante, g.modelo, memoria))
        linhas.append("")

    linhas.append("GPU disponivel para o framework: %s"
                  % ("SIM" if info.aceleracao_disponivel else "NAO"))
    if info.aceleracao_disponivel:
        for g in info.gpus_do_framework:
            linhas.append("GPU: %s" % g.nome)
            if g.memoria_total_bytes:
                linhas.append("VRAM: %.2f GB" % g.memoria_total_gb)
            if g.compute_capability:
                linhas.append("Compute capability: %s (bf16: %s)"
                              % (g.sm, "SIM" if g.suporta_bf16 else "NAO"))

    linhas.append("")
    linhas.append("Backend ativo: %s" % info.backend.upper())
    linhas.append("CUDA disponivel: %s" % ("SIM" if info.backend == "cuda" else "NAO"))
    linhas.append("CUDA (build do PyTorch): %s" % (info.build_cuda or "nenhum -- build CPU-only"))
    linhas.append("CUDA (suportado pelo driver): %s" % (info.cuda_do_driver or "n/a"))
    linhas.append("Driver da GPU: %s" % (info.versao_driver or "n/a"))
    linhas.append("cuDNN: %s" % (info.versao_cudnn or "n/a"))
    if info.arquiteturas_do_build:
        linhas.append("Arquiteturas no build: %s" % ", ".join(info.arquiteturas_do_build))

    linhas.append("")
    linhas.append("REQUIRE_GPU: %s" % ("SIM" if config.DEVICE_CFG.require_gpu else "NAO"))

    try:
        device = resolver_dispositivo(silencioso=True)
    except GPUIndisponivelError:
        device = None

    linhas.append("")
    linhas.append("Dispositivo selecionado:")
    if device is None:
        linhas.append("  NENHUM -- REQUIRE_GPU ativo e GPU indisponivel (o treino seria abortado)")
    else:
        linhas.append("  %s" % _rotulo_dispositivo(device))
        if device.type == "cuda":
            habilitado, dtype, motivo = configuracao_amp(device)
            linhas.append("")
            linhas.append("Mixed precision: %s (%s)"
                          % ("SIM" if habilitado else "NAO", motivo))
            linhas.append("Batch size: %d | num_workers: %d | pin_memory: %s"
                          % (config.DATA_CFG.batch_size, config.DATA_CFG.num_workers,
                             config.DATA_CFG.pin_memory))
            m = memoria_gpu(device)
            if m:
                linhas.append("Memoria da GPU agora: %.0f MB livres de %.0f MB"
                              % (m["livre"], m["total"]))

    linhas.append("")
    if info.problemas:
        linhas.append("Status: PROBLEMAS ENCONTRADOS")
        linhas.append("")
        linhas.append("O QUE PRECISA SER CORRIGIDO:")
        for p in info.problemas:
            linhas.append("  - " + p)
        for c in info.correcoes:
            linhas.append("  > " + c)
    elif device is not None and device.type != "cpu":
        linhas.append("Status: OK")
    else:
        linhas.append("Status: OK (sem GPU neste computador; treino em CPU)")

    linhas.append(_SEP_FORTE)
    return "\n".join(linhas)


def main() -> int:
    """Executa o diagnostico completo e devolve o codigo de saida."""
    info = obter_hardware()
    print(relatorio_diagnostico(info))

    if info.aceleracao_disponivel:
        print()
        try:
            device = resolver_dispositivo(silencioso=True)
            verificar_uso_real_da_gpu(device)
        except GPUIndisponivelError as exc:
            print(exc)
            return 1
        print(_SEP_FORTE)

    return 1 if info.problemas else 0


if __name__ == "__main__":
    sys.exit(main())
