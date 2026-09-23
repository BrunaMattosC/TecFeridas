# Segmentação e Medição Automática de Feridas

Sistema completo em PyTorch para segmentação semântica de feridas e
medição automática de suas características morfológicas (área,
comprimento, largura, perímetro, circularidade, solidez etc.), usando o
dataset
[Wound Segmentation Images](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images).

Fluxo completo:

```
imagem → segmentação (U-Net) → máscara binária → pós-processamento
       → componentes conectados → medição por ferida
       → conversão opcional para cm → relatório CSV + visualização anotada
```

**Estado verificado:** 54 testes automatizados passando; dataset de 2.760
pares processado ponta a ponta; 3.414 feridas medidas em 23 segundos.

---

## Índice

1. [Início rápido](#início-rápido)
2. [Tutorial passo a passo](#tutorial-passo-a-passo)
3. [Treinar na GPU](#treinar-na-gpu)
4. [Solução de problemas](#solução-de-problemas)
5. [Referência de comandos](#referência-de-comandos)
6. [Estrutura do projeto](#estrutura-do-projeto)
7. [Como a segmentação funciona](#como-a-segmentação-funciona)
8. [Medição (measure.py)](#medição-measurepy)
9. [Testes](#testes)
10. [Mudanças recentes](#mudanças-recentes)

---

## Início rápido

Para quem já tem Python 3.10+ instalado e só quer ver funcionando:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python verificar_ambiente.py       # diz o que falta nesta máquina
pytest tests/ -q                   # deve responder: 54 passed
```

Se `verificar_ambiente.py` terminar sem apontar pendências, o ambiente
está pronto. O tutorial abaixo cobre cada etapa em detalhe.

---

## Tutorial passo a passo

Este tutorial leva do zero até medir feridas reais. Ele foi escrito para
funcionar em **Windows, Linux e macOS**.

### Passo 0 — Pré-requisitos

| Requisito | Mínimo | Observação |
| --- | --- | --- |
| Python | **3.10** | O código usa a sintaxe `X \| None`. Verificado em 3.14.6. |
| Espaço em disco | ~2,5 GB | ~700 MB de dataset + ~1,5 GB de dependências. |
| Conta no Kaggle | — | Só para baixar o dataset. Gratuita. |
| GPU NVIDIA | opcional | Só acelera o **treino**. Precisa de *compute capability* ≥ 5.0. |

Confira sua versão do Python:

```bash
python --version
```

Se aparecer 3.9 ou inferior, instale uma versão mais nova em
[python.org/downloads](https://www.python.org/downloads/). Em alguns
sistemas o comando é `python3` em vez de `python`.

### Passo 1 — Obter o código

Copie a pasta do projeto para a máquina. **Não copie** os diretórios
gerados — eles somam mais de 1 GB e são todos recriáveis:

| Não copiar | Como recriar |
| --- | --- |
| `dataset/` | `python dataset.py` (~700 MB) |
| `checkpoints/` | `python train.py` |
| `outputs/`, `logs/` | Gerados a cada execução |
| `__pycache__/`, `.venv/` | Específicos da máquina e da versão do Python |

### Passo 2 — Criar o ambiente virtual

Isolar as dependências evita conflito com outros projetos da máquina.

```bash
python -m venv .venv
```

Ative conforme o sistema:

| Sistema | Comando |
| --- | --- |
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| Windows (Git Bash) | `source .venv/Scripts/activate` |

Com o ambiente ativo, o nome `(.venv)` aparece no início do *prompt*.

> **PowerShell bloqueando o script?** Rode antes, na mesma janela:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Passo 3 — Instalar as dependências

> **A escolha do índice do PyTorch decide se você treina em GPU ou em CPU.**
> `pip install torch` sozinho, ou com `--index-url .../whl/cpu`, instala uma
> build **sem nenhum suporte a GPU** (`torch.version.cuda is None`). Nessa
> build, nenhuma configuração de código faz a GPU funcionar — a única
> correção é reinstalar a partir do índice CUDA correto. Veja
> [Treinar na GPU](#treinar-na-gpu).

**Caso A — máquina com GPU NVIDIA** (*compute capability* ≥ 5.0, ou seja,
GTX 900 em diante). Escolha o índice pela versão de CUDA que **o seu driver**
suporta — leia-a no canto superior direito de `nvidia-smi`:

```bash
# Driver com CUDA 12.x (a maioria das instalações hoje)
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Driver com CUDA 13.x (exige driver >= 580)
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements-dev.txt
```

Instalar um wheel `cu130` com um driver de CUDA 12.x resulta em
`torch.cuda.is_available() == False`, sem erro de instalação. Na dúvida,
`python hardware.py` diz qual índice usar nesta máquina.

**Caso B — máquina sem GPU, ou com GPU antiga** (GTX 6xx/7xx e anteriores).
A variante CPU do PyTorch é bem menor e evita baixar CUDA à toa:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

Nesse caso, defina `REQUIRE_GPU=0` antes de treinar — caso contrário
`train.py` aborta de propósito, em vez de iniciar um treino em CPU que
levaria dias (ver [Treinar na GPU](#treinar-na-gpu)).

**Caso C — reproduzir exatamente o ambiente já validado:**

```bash
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-verificado.txt
```

> **Só quer medir máscaras, sem treinar nem prever?** A medição não depende
> de PyTorch. Bastam:
> `pip install opencv-python-headless numpy pandas matplotlib pytest`

### Passo 4 — Diagnosticar o ambiente

```bash
python verificar_ambiente.py
```

Este script roda **sem nenhuma dependência instalada** (usa só a
biblioteca padrão), então funciona até antes do primeiro `pip install`.
Ele verifica versão do Python, ambiente virtual, espaço em disco, cada
dependência, GPU, credenciais do Kaggle, dataset e checkpoints — e
termina dizendo o que a máquina consegue fazer **agora**:

```
------------------------------------------------------------------------
VEREDITO
------------------------------------------------------------------------
[ OK   ] Medir mascaras (measure.py)      : SIM
[ OK   ] Rodar a suite de testes          : SIM
[ OK   ] Baixar/explorar o dataset        : SIM
[ OK   ] Treinar (train.py)               : SIM
[FALTA ] Prever/avaliar em imagens        : NAO

PROXIMOS PASSOS, NESTA ORDEM:
  1. python train.py                   (gera um checkpoint)
```

Código de saída: `0` = pronto para ao menos a medição; `1` = falta algo
essencial. Isso permite usá-lo em scripts de CI.

### Passo 5 — Configurar as credenciais do Kaggle

Necessário **apenas para baixar o dataset**. Gere o token em
[kaggle.com/settings](https://www.kaggle.com/settings) e use **uma** das
três opções:

```bash
# Opção 1 — token moderno (formato KGAT_...), exige kaggle >= 2.0
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxx

# Opção 2 — usuário e chave
export KAGGLE_USERNAME=seu_usuario
export KAGGLE_KEY=sua_chave

# Opção 3 — arquivo padrão da API
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

No Windows (PowerShell), a sintaxe é:

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxx"
```

> **Nunca** grave credenciais dentro do projeto. Variáveis de ambiente
> definidas assim valem só para a janela de terminal atual — para torná-las
> permanentes, use as configurações do sistema.

### Passo 6 — Baixar o dataset

```bash
python dataset.py
```

O script autentica, baixa (~665 MB), descompacta, organiza as imagens em
`dataset/images/` e as máscaras em `dataset/masks/`, e salva um CSV de
exploração. É **idempotente**: se os dados já existem, o download é
pulado.

O que você deve obter:

| Item | Valor esperado |
| --- | --- |
| Pares imagem/máscara | 2.760 |
| Resolução | 512×512 (uniforme) |
| Fontes | `fusc` (1.210), `wsnet` (1.176), `medetec` (374) |
| Máscaras vazias | 24 (imagens sem ferida anotada) |
| Ferida mediana | 1,55% da área da imagem |

Para gerar os gráficos de exploração:

```bash
python explore_data.py
```

### Passo 7 — Validar a instalação

```bash
pytest tests/ -q
```

Resultado esperado:

```
54 passed
```

Se aparecer `45 passed, 1 skipped`, a pilha de deep learning não está
instalada — os testes de medição rodam mesmo assim, mas os 9 de
integração são pulados. Volte ao Passo 3 se pretende treinar ou prever.

### Passo 8 — Medir sem modelo treinado

A medição opera sobre **qualquer máscara binária**, inclusive as máscaras
verdadeiras do dataset. Isso permite ver o sistema funcionando antes de
treinar qualquer coisa — e, de quebra, isola a qualidade da *medição* da
qualidade da *segmentação*.

O script `exemplo_medicao.py` faz exatamente isso:

```bash
python exemplo_medicao.py                    # escolhe uma imagem automaticamente
python exemplo_medicao.py medetec_0153.png   # imagem específica
python exemplo_medicao.py medetec_0153.png --sem-rotulos
```

Saída:

```
============================================================
Imagem : medetec_0153.png
Feridas: 8
============================================================
Ferida 1
  Área: 6779 px²
  Comprimento: 207.1 px
  Largura: 89.2 px
  ...

Imagem anotada: outputs/predictions/exemplo_medicao/medetec_0153_medicoes.png
Relatorio CSV : outputs/predictions/exemplo_medicao/medetec_0153_medicoes.csv
```

Essa imagem tem **8 feridas distintas**, todas medidas individualmente.
Use `--sem-rotulos` quando houver muitas feridas: os blocos de texto se
sobrepõem e a anotação fica ilegível.

Na imagem anotada, as cores significam:

| Cor | Elemento |
| --- | --- |
| Verde | Contorno da ferida |
| Amarelo | Bounding box alinhada aos eixos |
| Ciano | Retângulo de área mínima (de onde saem comprimento e largura) |
| Vermelho | Eixo de comprimento |
| Azul | Eixo de largura |

**Desempenho medido:** 2.760 máscaras de 512×512 medidas em 23,1 s
(119 imagens/s, ~8 ms por imagem) em 4 núcleos de CPU.

### Passo 9 — Treinar

```bash
python train.py
```

Arquitetura, encoder e hiperparâmetros ficam em `config.py`
(`ModelConfig` / `TrainConfig`). Para trocar, edite
`MODEL_CFG.architecture` (`"unet"`, `"unetplusplus"`, `"deeplabv3plus"`,
`"fpn"`) e `MODEL_CFG.encoder_name` (`"resnet34"`, `"efficientnet-b0"`,
`"efficientnet-b3"`).

Saídas:

- `checkpoints/best_<run>.pt` — melhor checkpoint (pelo Dice de validação);
- `outputs/reports/training_history_<run>.csv` — métricas por época;
- `logs/tensorboard/<run>` — visualize com `tensorboard --logdir logs/tensorboard`.

O treino usa a **GPU automaticamente** quando há uma disponível, e imprime
o hardware escolhido, a memória e a ocupação da placa a cada época. Veja
[Treinar na GPU](#treinar-na-gpu) para o detalhamento.

> **Atenção ao tempo.** Numa RTX 3050 6 GB Laptop, as 60 épocas levam
> **cerca de 1,9 h**. **Em CPU é impraticável**: o mesmo treino levaria
> ~49 h (medido, mesma máquina). Por isso `train.py` **aborta** se a GPU
> não estiver disponível, em vez de começar um treino de dois dias sem
> avisar — use `REQUIRE_GPU=0` para forçar CPU conscientemente. Sem GPU,
> prefira o [Google Colab](#google-colab), que oferece GPU gratuita, ou
> reduza `DATA_CFG.image_size` e `TRAIN_CFG.epochs` em `config.py` para
> gerar um modelo apenas demonstrativo.

### Passo 10 — Avaliar no conjunto de teste

```bash
python evaluate.py --checkpoint checkpoints/best_<run>.pt
```

Gera matriz de confusão, curva ROC, curva Precision-Recall e os
relatórios por imagem e por ferida em `outputs/reports/`.

### Passo 11 — Segmentar e medir uma imagem

```bash
python predict.py \
    --image caminho/para/imagem.png \
    --checkpoint checkpoints/best_<run>.pt \
    --pixels-por-cm 38.0   # opcional; se omitido, as medidas ficam em pixels
```

Salva em `outputs/predictions/`:

- `<nome>_comparacao.png` — original / máscara real (se houver) / prevista / overlay / **medições**;
- `<nome>_overlay.png`;
- `<nome>_medicoes.png` — imagem anotada com contorno, bounding box, retângulo mínimo, eixos de comprimento/largura e os valores;
- `<nome>_medicoes.csv` — uma linha por ferida detectada.

### Passo 12 — Inferência em lote

```bash
python batch_inference.py \
    --input-dir caminho/da/pasta \
    --checkpoint checkpoints/best_<run>.pt \
    --output-dir outputs/predictions/lote \
    --pixels-por-cm 38.0
```

Além dos artefatos por imagem, gera dois CSVs consolidados:
`medicoes_lote.csv` (uma linha por imagem, ferida principal) e
`medicoes_por_ferida.csv` (uma linha por ferida detectada).

Se uma imagem falhar, o erro é registrado em log e o lote continua.

---

## Treinar na GPU

O projeto usa a GPU automaticamente quando há uma disponível. Esta seção
explica como conferir isso, o que muda em relação à CPU e o que fazer
quando a GPU existe mas não está sendo usada.

### Conferir antes de treinar

```bash
python hardware.py
```

O script detecta o sistema operacional, **todas** as placas de vídeo
instaladas (sem assumir fabricante — NVIDIA, AMD, Intel e Apple são
enumeradas), o driver, a versão de CUDA suportada pelo driver, a versão do
PyTorch e a de CUDA com que ele foi compilado, o cuDNN, a *compute
capability* e o suporte a bf16. Depois **executa uma operação de verdade na
GPU** e mede seu desempenho — a configuração só é considerada boa se a
placa realmente processou a conta.

Saída em uma máquina configurada:

```
==================================================================
DIAGNOSTICO DO AMBIENTE
==================================================================

Python: 3.13.9
Sistema: Windows 11 (AMD64)
Framework: PyTorch 2.13.0+cu126

GPUs instaladas no sistema:
  - [NVIDIA] NVIDIA GeForce RTX 3050 6GB Laptop GPU
  - [Intel] Intel(R) UHD Graphics

GPU disponivel para o framework: SIM
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM: 6.00 GB
Compute capability: sm_86 (bf16: SIM)

Backend ativo: CUDA
CUDA disponivel: SIM
CUDA (build do PyTorch): 12.6
CUDA (suportado pelo driver): 12.7
Driver da GPU: 566.36
cuDNN: 9.10.2

Dispositivo selecionado:
  GPU

Status: OK
==================================================================
```

Se algo estiver errado, o relatório diz **exatamente** o quê e como
corrigir — inclusive qual índice do PyTorch instalar, já ajustado ao driver
que a máquina tem. `python verificar_ambiente.py` mostra um resumo dessas
mesmas informações junto com o restante do diagnóstico.

### O que aparece durante o treino

`train.py` imprime o hardware escolhido, verifica a GPU antes da primeira
época e, a cada época, reporta memória, ocupação e tempo:

```
Epoch 2/60
Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
GPU Memory Allocated: 388 MB
GPU Memory Reserved: 2690 MB
Training...
Train Loss: 0.42820
Val Loss: 0.35660
Val Dice: 0.75330
GPU Memory: 388 MB alocada / 2690 MB reservada / pico 2157 MB
GPU Usage: media 97% | pico 100%
Time: 114.2s (0.436s/batch no treino)
```

A **ocupação média** é medida por amostragem contínua da NVML enquanto a
época roda, em uma thread separada. É a evidência de que a GPU está de
fato trabalhando — diferente de apenas imprimir `device = cuda`.

> **A primeira época é sempre mais lenta.** No Windows, cada *worker* do
> DataLoader é um processo novo que precisa reimportar o PyTorch; com 8
> workers isso custa ~2 minutos, pagos **uma vez** graças a
> `persistent_workers`. Na medição abaixo, a época 1 levou 236 s e a
> época 2, já em regime, 114 s — com a ocupação da GPU subindo de 63%
> para 97%. Não confunda esse custo de partida com "a GPU não está sendo
> usada".

### Desempenho medido

UNet + ResNet34 @ 512×512, batch 8, AMP fp16, nas 2.761 imagens do
dataset (1.932 de treino / 414 de validação):

| | CPU (i7-13620H) | GPU (RTX 3050 6 GB Laptop) |
|---|---|---|
| Tempo por batch (treino) | 12,11 s | **0,436 s** |
| Tempo por época de treino | ~49 min | **105 s** |
| Época completa (treino + validação) | — | **114 s** |
| 60 épocas | ~49 h | **~1,9 h** |
| Ocupação média da GPU | — | **97%** |
| Pico de VRAM | — | 2.157 MB de 6.144 MB (35%) |

A precisão mista sozinha responde por parte do ganho: no mesmo lote,
**fp16 é 1,61× mais rápido que fp32 e usa 1,8 GB a menos de VRAM**
(0,393 s/batch contra 0,631 s/batch).

**Por que o batch size continua 8.** Medindo o pico de VRAM real com o
modelo do projeto: batch 8 usa 2.240 MB (36% dos 6 GB) e já satura o
throughput — 0,048 s por imagem, o mesmo que batch 12 e batch 16. Aumentar
o lote gastaria memória sem ganhar velocidade, e mudaria o ruído do
gradiente. Por isso **nenhum hiperparâmetro de treino foi alterado**.

Tudo isso também vai para o CSV de histórico
(`outputs/reports/training_history_<run>.csv`, colunas `device`,
`tempo_epoca_s`, `gpu_memoria_*`, `gpu_utilizacao_media_percent`) e para o
TensorBoard, o que permite comparar objetivamente o custo por época em CPU
e em GPU.

### Sem fallback silencioso

Se a máquina tem GPU e o PyTorch não consegue usá-la, o projeto **não**
começa a treinar em CPU calado. Ele avisa, dizendo o motivo:

```
==================================================================
ATENCAO
A GPU foi detectada, porem NAO esta sendo utilizada pelo treinamento.
Verifique a instalacao/configuracao do ambiente.
------------------------------------------------------------------
  MOTIVO : GPU presente (NVIDIA GeForce RTX 3050 6GB Laptop GPU), mas o
           PyTorch instalado (2.13.0+cpu) e uma build SOMENTE CPU -- ele
           foi compilado sem suporte a GPU.
  CORRIJA: Reinstale o PyTorch com suporte a GPU:
           pip install --index-url https://download.pytorch.org/whl/cu126 \
               torch==2.13.0 torchvision
==================================================================
```

Com `REQUIRE_GPU` ligado (o padrão), `train.py` **aborta** com código de
saída `2` em vez de iniciar um treino de dias em CPU. Avaliação e
inferência continuam funcionando em CPU — elas apenas avisam.

Para treinar em CPU deliberadamente:

```bash
# Windows (PowerShell)
$env:REQUIRE_GPU = "0";  python train.py
# Linux / macOS
REQUIRE_GPU=0 python train.py
```

### Configuração

Tudo vive em `config.DEVICE_CFG` (classe `DeviceConfig` em `config.py`) e
pode ser sobrescrito por variável de ambiente, sem editar código:

| Variável | Padrão | Efeito |
|---|---|---|
| `REQUIRE_GPU` | `1` | Aborta o treino se a GPU não estiver disponível |
| `DEVICE` | `auto` | `cpu` força CPU (útil para medir CPU × GPU) |
| `GPU_INDEX` | `0` | Qual placa usar, em máquinas com mais de uma |
| `AMP_DTYPE` | `fp16` | `bf16` (Ampere+, dispensa GradScaler) ou `auto` |
| `ALLOW_TF32` | `1` | TF32 em matmul/convolução (Ampere+) |
| `CUDNN_BENCHMARK` | `0` | Ver *Reprodutibilidade*, abaixo |
| `BATCH_SIZE` | `8` | Tamanho do lote |
| `NUM_WORKERS` | metade dos núcleos, máx. 8 | Processos de carregamento |
| `LOG_GPU_EVERY_N_BATCHES` | `0` | Log de memória a cada N batches |

O ponto de decisão é único: `hardware.resolver_dispositivo()`. Treino,
validação, teste, avaliação, geração de máscaras, inferência em uma imagem
e inferência em lote chamam essa mesma função — não existe outro lugar no
código que escolha entre GPU e CPU.

### O que a mudança para GPU alterou — e o que não alterou

**Não** foram alterados: arquitetura do modelo, dataset, máscaras, divisão
treino/validação/teste (70/15/15, seed 42), função de perda, métricas,
número de épocas, critérios de avaliação e pré-processamento. O modelo
treinado na GPU é o mesmo modelo, treinado mais rápido.

Alterações feitas **especificamente** por causa da GPU, todas de execução e
não de metodologia:

- **Precisão mista (AMP).** `TRAIN_CFG.use_amp` já era `True`, mas nunca
  tinha efeito, porque a build instalada era CPU-only. Agora a *forward* do
  modelo roda em fp16 nos Tensor Cores.
  A **loss é calculada fora do `autocast`, em float32**, de propósito:
  `DiceLoss` soma as probabilidades dos 512×512 = 262.144 pixels da imagem,
  e o maior valor finito em fp16 é 65.504 — a soma saturaria em `+inf` e a
  loss viraria `NaN` assim que ~25% da imagem tivesse probabilidade alta.
  Promovendo os logits a float32 antes da loss, o valor calculado é o mesmo
  de um treino em precisão simples. Métricas e curvas ROC/PR seguem a mesma
  regra.
- **TF32** ligado em matmul/convolução (Ampere+): o acumulador continua em
  fp32, então o impacto numérico é muito menor que o do fp16.
- **DataLoader:** `pin_memory` (habilita a cópia assíncrona host→device),
  `persistent_workers` (no Windows, recriar processos a cada época custa
  segundos), `prefetch_factor=4` e `drop_last` **apenas no treino** — em
  validação e teste ele continua `False`, para não descartar amostras de
  avaliação. Nenhum desses parâmetros muda quais amostras chegam ao modelo
  nem em que ordem.
- **`num_workers`** passou de `min(4, núcleos)` para metade dos núcleos
  lógicos, limitado a 8: em GPU o gargalo deixa de ser a rede e passa a ser
  a augmentation em CPU.

### Reprodutibilidade

`utils.seed_everything` continua fixando `cudnn.deterministic = True` e
`cudnn.benchmark = False`, exatamente como antes. O `cuDNN benchmark`
(que testa algoritmos de convolução e fixa o mais rápido) aceleraria mais,
mas é incompatível com determinismo — por isso vem **desligado**. Ligue
conscientemente com `CUDNN_BENCHMARK=1` se aceitar trocar reprodutibilidade
por velocidade.

### `CUDA out of memory`

Reduza o lote sem editar código:

```bash
$env:BATCH_SIZE = "4";  python train.py     # Windows
BATCH_SIZE=4 python train.py                # Linux / macOS
```

Se ainda faltar memória, reduza `DATA_CFG.image_size` em `config.py` — mas
note que isso **muda o pré-processamento** e, portanto, os resultados; a
comparação com execuções anteriores deixa de ser direta.

---

## Solução de problemas

| Sintoma | Causa | Solução |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'cv2'` | Dependências não instaladas ou ambiente virtual não ativado | Ative o `.venv` e rode `pip install -r requirements-dev.txt` |
| `AttributeError: module 'numpy' has no attribute 'trapz'` | Versão antiga do projeto com NumPy 2.x | Já corrigido nesta versão. Se persistir, confirme que `evaluate.py` usa `sklearn.metrics.auc` |
| `RuntimeError: get_train_transforms() exige 'albumentations' e 'torch'` | Ambiente mínimo, sem a pilha de deep learning | `pip install -r requirements.txt` |
| `RuntimeError: Falha ao autenticar na API do Kaggle` | Credenciais ausentes ou inválidas | Ver [Passo 5](#passo-5--configurar-as-credenciais-do-kaggle) |
| `FileNotFoundError: Nenhuma imagem/máscara encontrada` | Dataset não baixado | `python dataset.py` |
| Treino não usa a GPU | PyTorch instalado na variante CPU (`torch.version.cuda is None`), wheel CUDA mais novo que o driver, ou placa com *compute capability* < 5.0 | Rode `python hardware.py` — ele diz o motivo exato e o comando de correção. Ver [Treinar na GPU](#treinar-na-gpu) |
| `TREINAMENTO ABORTADO: GPU EXIGIDA E INDISPONÍVEL` | `REQUIRE_GPU` ligado (padrão) e GPU inutilizável — proteção contra treino acidental em CPU | Corrija a GPU conforme a linha acima, ou `REQUIRE_GPU=0` para treinar em CPU de propósito |
| `CUDA out of memory` | `batch_size` alto demais para a VRAM | `BATCH_SIZE=4 python train.py` (ou `$env:BATCH_SIZE="4"` no PowerShell) |
| Primeira época muito mais lenta que as seguintes | Spawn dos *workers* do DataLoader no Windows (~2 min com 8 workers) | Normal e pago uma vez só. Reduza `NUM_WORKERS` se a RAM for curta |
| `UnicodeEncodeError` no console do Windows | Console em cp1252 | `chcp 65001` antes de rodar. `verificar_ambiente.py` já usa só ASCII e não é afetado |
| PowerShell recusa ativar o `.venv` | Política de execução | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Treino travado ou lentíssimo | Rodando em CPU | Esperado. Ver aviso no [Passo 9](#passo-9--treinar) |
| `pytest` não encontra os módulos | Executado de fora da raiz do projeto | Rode `pytest` a partir da pasta do projeto |

Em qualquer dúvida sobre o estado da máquina, o primeiro comando é sempre:

```bash
python verificar_ambiente.py
```

---

## Referência de comandos

| Objetivo | Comando |
| --- | --- |
| Diagnosticar o ambiente | `python verificar_ambiente.py` |
| Diagnosticar GPU / dispositivo | `python hardware.py` |
| Instalar (execução) | `pip install -r requirements.txt` |
| Instalar (execução + testes) | `pip install -r requirements-dev.txt` |
| Instalar versões exatas validadas | `pip install -r requirements-verificado.txt` |
| Rodar os testes | `pytest tests/ -q` |
| Baixar e organizar o dataset | `python dataset.py` |
| Gerar gráficos de exploração | `python explore_data.py` |
| Medir sem modelo treinado | `python exemplo_medicao.py [imagem.png] [--sem-rotulos]` |
| Treinar | `python train.py` |
| Avaliar | `python evaluate.py --checkpoint checkpoints/best_<run>.pt` |
| Prever (medidas em pixels) | `python predict.py --image IMG --checkpoint CKPT` |
| Prever (escala direta) | `python predict.py --image IMG --checkpoint CKPT --pixels-por-cm 38.0` |
| Prever (objeto de referência) | `python predict.py --image IMG --checkpoint CKPT --referencia-pixels 190 --referencia-cm 5.0` |
| Inferência em lote | `python batch_inference.py --input-dir DIR --checkpoint CKPT` |
| TensorBoard | `tensorboard --logdir logs/tensorboard` |

---

## Estrutura do projeto

```
project/
├── dataset/                 # imagens/máscaras organizadas (gerado automaticamente)
│   ├── raw/                  # download bruto do Kaggle
│   ├── images/
│   └── masks/
├── outputs/
│   ├── figures/              # histogramas, curvas, matriz de confusão...
│   ├── reports/              # CSVs (exploração, histórico, relatórios por imagem/ferida)
│   └── predictions/          # máscaras/overlays/medições de predict.py e batch_inference.py
├── checkpoints/              # melhores pesos salvos durante o treino
├── logs/                     # logs em texto + TensorBoard
├── tests/                    # testes automatizados (pytest)
│   ├── conftest.py            # máscaras sintéticas de referência
│   ├── test_measure.py        # medição (só precisa de opencv + numpy)
│   └── test_pipeline_integration.py  # segmentação -> medição ponta a ponta
├── verificar_ambiente.py     # diagnóstico: o que está pronto e o que falta nesta máquina
├── exemplo_medicao.py        # demonstra a medição sem exigir modelo treinado
├── config.py                 # configuração central (paths, hiperparâmetros, dispositivo, medição)
├── hardware.py               # detecção de GPU, escolha de dispositivo, AMP, monitoramento
├── utils.py                  # seeds, logging, device (atalho p/ hardware.py), visualização
├── dataset.py                # download Kaggle, exploração, Dataset PyTorch, split, augmentations
├── explore_data.py           # Etapa 2: exploração e gráficos do dataset
├── models.py                 # Etapa 4: fábrica de modelos (Unet/Unet++/DeepLabV3+/FPN)
├── losses.py                 # Etapa 5: Dice Loss + BCE
├── metrics.py                # Etapa 6: Dice, IoU, Precision, Recall, F1, Accuracy
├── train.py                  # Etapa 7: loop de treinamento completo
├── predict.py                # Etapa 8: predição em uma imagem + medição + visualização
├── measure.py                # Etapas 9-11: medição morfológica + escala + relatório + anotação
├── evaluate.py               # Etapas 12-13: avaliação no teste + curvas/matriz de confusão
├── batch_inference.py        # Etapa 15: inferência em lote sobre uma pasta
├── requirements.txt              # dependências de execução (faixas com limite superior)
├── requirements-dev.txt          # execução + pytest
└── requirements-verificado.txt   # versões exatas com a suíte validada
```

### Política de versões

`requirements.txt` usa faixas com **limite superior** (`>=x,<y`), não
apenas `>=`. Isso é deliberado: a versão anterior permitia qualquer versão
futura, e o NumPy 2.0 **removeu** `np.trapz`, quebrando `evaluate.py` em
toda instalação nova. Com limite superior, uma major nova não entra
sozinha e derruba o projeto.

Para atualizar uma faixa: suba o limite, rode `pytest tests/ -q` e só
então registre a mudança.

---

## Como a segmentação funciona

- **Modelo:** U-Net (`segmentation_models_pytorch`) com encoder
  ResNet34 pré-treinado no ImageNet. Arquitetura e encoder são
  trocáveis em `config.ModelConfig` (`unet`, `unetplusplus`,
  `deeplabv3plus`, `fpn`).
- **Entrada:** imagem RGB de qualquer resolução. É redimensionada para
  `DATA_CFG.image_size` (512×512) e normalizada com média/desvio do
  ImageNet (`dataset.get_eval_transforms`).
- **Saída:** um único canal de *logits*. Aplica-se `sigmoid` e o limiar
  de 0,5, produzindo uma **máscara binária de uma classe**
  (ferida × fundo) — não há bounding boxes nem polígonos nativos.
- **Reescala:** a máscara volta à resolução **original** da imagem com
  interpolação `INTER_NEAREST` (`predict.predict_mask`), de modo que as
  medidas em pixels se referem sempre à imagem original.

---

## Medição (measure.py)

A medição parte **exclusivamente da máscara realmente produzida pela
segmentação** — nada é estimado visualmente a partir da foto.

### Pós-processamento da máscara

Antes de medir, `measure.clean_mask()` aplica (configurável em
`config.MeasurementConfig`):

1. **Preenchimento de buracos** (`fill_holes=True`) — regiões internas
   classificadas como fundo dentro da ferida (tecido de granulação,
   fibrina) são reincorporadas. Altera apenas o interior; a borda
   externa fica intacta.
2. **Remoção de componentes pequenos** — descarta regiões com área
   menor que `max(min_component_area_px, min_component_area_ratio × área da imagem)`
   (padrão: `max(50 px, 0,02%)`). Se *todos* os componentes ficarem
   abaixo do limiar, o maior é preservado para não descartar uma ferida
   legítima em imagem de baixa resolução.
3. **Abertura/fechamento morfológico** — **desativados por padrão**
   (`opening_kernel_size = closing_kernel_size = 0`). Operações
   morfológicas deslocam a borda e, portanto, enviesam justamente a
   área e o perímetro que se quer medir (uma abertura 3×3 elíptica
   remove os cantos de um retângulo). A remoção de ruído é feita pelo
   filtro de área, que é exato e preserva a borda. Ative-as apenas se
   as máscaras do seu modelo vierem serrilhadas.

Cada componente conectado sobrevivente (conectividade 8) é tratado como
uma **ferida independente**, recebendo um `wound_id` sequencial em ordem
decrescente de área.

### Métricas calculadas

| Métrica | Campo | Como é calculada |
| --- | --- | --- |
| Área | `area_pixels` | contagem exata de pixels do componente |
| Perímetro | `perimeter_pixels` | `cv2.arcLength` sobre o contorno externo completo (`CHAIN_APPROX_NONE`) |
| Comprimento | `length_pixels` | **maior lado** do retângulo de área mínima (`cv2.minAreaRect`) |
| Largura | `width_pixels` | **menor lado** do mesmo retângulo (perpendicular ao comprimento) |
| Bounding box | `bounding_box` | `(x, y, w, h)` alinhada aos eixos da imagem |
| Centroide | `centroid` | momentos de imagem (`m10/m00`, `m01/m00`) |
| Diâmetro de Feret máx. | `feret_max_pixels` | maior distância entre dois pontos do fecho convexo |
| Diâmetro equivalente | `equivalent_diameter_pixels` | `sqrt(4·área/π)` — diâmetro do círculo de mesma área |
| Razão comprimento/largura | `length_width_ratio` | `length / width` |
| Razão de aspecto | `aspect_ratio` | `w / h` da bounding box (alinhada aos eixos) |
| Circularidade | `circularity` | `4π·área / perímetro²` (1 = círculo perfeito) |
| Solidez | `solidity` | `área / área do fecho convexo` |
| Extensão | `extent` | `área / área da bounding box` |
| Orientação | `orientation_deg` | ângulo do eixo de maior comprimento, em `[0, 180)` |
| Área convexa | `convex_area_pixels` | `cv2.contourArea` do fecho convexo |
| % da imagem | `percent_image_area` | `100 × área / (H·W)` |
| Nº de componentes | `num_connected_components` | componentes conectados na imagem |

**Por que comprimento/largura vêm do `minAreaRect` e não da bounding
box?** A bounding box é alinhada aos eixos da imagem: uma ferida
inclinada 45° produziria uma caixa muito maior que a ferida. O
retângulo de área mínima é invariante à rotação e reproduz a convenção
clínica de "maior comprimento × maior largura perpendicular".

### Unidades: pixels ou centímetros?

**Por padrão, todas as medidas saem em pixels e px².** O dataset
utilizado não contém informação de escala (nem EXIF confiável, nem
marcador de referência, nem distância câmera-ferida), e o projeto
**nunca infere um fator de conversão**: `config.DEFAULT_PIXELS_POR_CM`
é `None` e todos os campos `*_cm` ficam `None`.

Para obter medidas físicas reais é preciso fornecer explicitamente a
escala. A classe `measure.PixelScale` centraliza a conversão:

```python
from measure import PixelScale, measure_wounds

PixelScale.from_pixels_per_cm(38.0)          # fator direto
PixelScale.from_pixels_per_mm(3.8)           # equivalente, em mm
PixelScale.from_reference_object(190.0, 5.0) # objeto de 5 cm medindo 190 px
PixelScale.from_dpi(300.0)                   # imagens de escâner

wounds = measure_wounds(mask, pixels_por_cm=PixelScale.from_pixels_per_cm(38.0))
```

Pela linha de comando (`predict.py` e `batch_inference.py`):

```bash
--pixels-por-cm 38.0
--pixels-por-mm 3.8
--referencia-pixels 190 --referencia-cm 5.0
```

Conversões aplicadas: comprimentos `px / pixels_por_cm`; áreas
`px² / pixels_por_cm²`. Helpers para milímetros
(`length_to_mm`, `area_to_mm2`) já existem em `PixelScale`, de modo que
adicionar outras unidades não exige alterar a medição.

**Melhor abordagem recomendada para escala física:** fotografar a ferida
com um marcador de tamanho conhecido (régua descartável, adesivo
circular ou etiqueta padronizada) posicionado **no mesmo plano da
ferida** e o mais próximo possível dela, com a câmera perpendicular à
superfície. Mede-se o marcador na imagem (em pixels) e passa-se
`--referencia-pixels` / `--referencia-cm`.

### APIs

```python
from measure import measure_wound, measure_wounds

measure_wounds(mask, pixels_por_cm=None)  # lista com UMA medição por ferida
                                          # [] quando não há ferida
measure_wound(mask, pixels_por_cm=None)   # API histórica: apenas o maior
                                          # componente, sem pós-processamento;
                                          # levanta ValueError se a máscara é vazia
```

`measure_wound` foi mantida com comportamento **idêntico** ao original
(verificado por teste diferencial) para não quebrar `evaluate.py`,
`batch_inference.py` nem código externo que já a utilize.

Nas 2.760 imagens do dataset, `measure_wounds` encontrou **3.414 feridas**,
enquanto `measure_wound` reportaria apenas 2.736 — a diferença de 678
feridas (19,9%) é o que a API antiga descartaria silenciosamente.

### Saída em texto

```python
from measure import format_measurements_text
print(format_measurements_text(wounds))
```

```text
Ferida 1
  Área: 28869 px²
  Comprimento: 260.3 px
  Largura: 140.4 px
  Perímetro: 678.2 px
  ...

Sem fator de escala informado: todas as medidas estão em pixels
(use --pixels-por-cm / --referencia-pixels + --referencia-cm).
```

### Limitações conhecidas

- **Sem escala física por padrão.** Sem um marcador de referência ou um
  fator informado manualmente, só há medidas relativas (pixels).
- **Perspectiva e distância não são corrigidas.** Fotos oblíquas
  encurtam a ferida na direção da inclinação; o fator de escala vale
  apenas para o plano do marcador de referência.
- **Superfícies curvas** (calcanhar, panturrilha) são medidas como
  projeção plana, subestimando a área real.
- **Perímetro levemente superestimado.** `arcLength` percorre a borda em
  escada pixel a pixel; para bordas suaves o erro é de alguns por cento,
  e a circularidade fica correspondentemente subestimada.
- **Circularidade e solidez podem passar de 1** em componentes pequenos:
  a área é contagem de pixels e o perímetro/fecho convexo vêm da
  geometria do contorno. Nas 3.414 feridas reais medidas, os máximos
  observados foram 1,35 e 1,48 respectivamente.
- **Feridas satélite muito próximas** podem ser fundidas em um único
  componente conectado; feridas separadas por menos de 1 px na máscara
  não são distinguidas.
- **Rótulos se sobrepõem** na imagem anotada quando há muitas feridas —
  `_draw_label_block()` mantém cada rótulo dentro da imagem, mas não
  trata colisão entre rótulos. Use `draw_labels=False` nesses casos.
- **Baixa resolução** aumenta o erro relativo: a medição é confiável
  apenas até a granularidade do pixel.
- **`max_wounds` (padrão 10)** limita quantas feridas são reportadas por
  imagem; o total detectado continua em `num_connected_components` e o
  truncamento é registrado em log.
- **A divisão oficial do dataset não é preservada.** O dataset traz
  `train_`/`test_` separados, mas `_organize_folders()` achata tudo e
  `split_dataset()` refaz um sorteio 70/15/15 sobre o conjunto inteiro.
  As métricas obtidas **não são comparáveis** com as de trabalhos que
  usam o split oficial.

---

## Testes

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

Resultado esperado: **54 passed**.

- `tests/test_measure.py` (45 testes) exercita a medição com máscaras
  sintéticas de geometria conhecida (retângulo, círculo, "L", retângulo
  rotacionado, máscara ruidosa, máscara vazia) e roda **sem** a pilha de
  deep learning — bastam `opencv` + `numpy` + `pandas`.
- `tests/test_pipeline_integration.py` (9 testes) valida o fluxo completo
  segmentação → medição usando um segmentador determinístico
  (`_StubSegmenter`), sem exigir um checkpoint treinado. É pulado
  automaticamente se `torch`/`smp`/`albumentations` não estiverem
  instalados.

Cenários cobertos: uma ferida, múltiplas feridas, formato irregular,
ruído/buracos na máscara, invariância à rotação, ausência de escala
física, ausência total de ferida, máscara de baixa resolução e
compatibilidade com a API antiga.

Os valores esperados são derivados **analiticamente** da geometria (um
retângulo 120×60 tem exatamente 7.200 px²), e não capturados de uma
execução anterior — são testes de correção, não de regressão.

---

## Google Colab

Todos os scripts funcionam sem alterações no Colab, que é a alternativa
recomendada para treinar quando a máquina local não tem GPU compatível:

```python
!pip install -r requirements.txt
import os; os.environ["KAGGLE_API_TOKEN"] = "KGAT_..."
!python dataset.py
!python train.py
```

O código detecta automaticamente a GPU quando disponível
(`hardware.resolver_dispositivo()`) e imprime qual dispositivo escolheu.
Se o runtime do Colab estiver sem acelerador, `train.py` **aborta** em vez
de treinar em CPU — troque para um runtime com GPU em *Ambiente de
execução → Alterar o tipo de ambiente de execução*, ou defina
`os.environ["REQUIRE_GPU"] = "0"` para treinar em CPU de propósito.

---

## Reprodutibilidade

`config.SEED` (padrão `42`) é usado por `utils.seed_everything()` para
fixar as seeds de Python, NumPy e PyTorch (CPU/GPU) e para garantir
que o split treino/validação/teste seja sempre o mesmo entre execuções.

---

## Mudanças recentes

### Correções de compatibilidade

- **`evaluate.py` estava quebrado em instalações novas.** Usava
  `np.trapz`, que foi **removida** no NumPy 2.0 — e `requirements.txt`
  pedia `numpy>=1.26.0` sem limite superior, então toda instalação a
  partir de 2024 falhava ao gerar a curva ROC. Substituído por
  `sklearn.metrics.auc`, que faz a mesma integração trapezoidal, já era
  dependência do projeto e independe da versão do NumPy.
- **`train.py` usa AMP compatível com as duas gerações do PyTorch.** As
  funções `hardware.criar_grad_scaler()` e `autocast_context()` escolhem
  entre `torch.amp` (2.4+) e `torch.cuda.amp` (anterior) em tempo de
  execução. Verificado em torch 2.13: zero avisos de depreciação.
- **`dataset.py` não exige mais a pilha de deep learning para baixar
  dados.** Os imports de `torch` e `albumentations` passaram a ser
  tolerantes (`DEEP_LEARNING_DISPONIVEL`). Download, organização,
  pareamento, divisão e exploração funcionam só com `cv2`/`numpy`/`pandas`;
  as funções que realmente precisam da pilha falham com mensagem
  explicativa.

### Dependências

- `requirements.txt` passou a usar **faixas com limite superior**.
- `scikit-image` foi **removida** — estava declarada, mas nenhum módulo
  do projeto a importava.
- `kaggle` subiu para `>=2.0`, mínimo para aceitar o token moderno.
- Novo `requirements-verificado.txt` com as versões exatas validadas.

### Aceleração por GPU

O treino passou a usar a GPU da máquina. Detalhes completos em
[Treinar na GPU](#treinar-na-gpu); o resumo:

- **A causa raiz era o ambiente, não o código.** O `torch` instalado era
  `2.13.0+cpu` — uma build **compilada sem suporte a GPU**
  (`torch.version.cuda is None`). O código já pedia `device = "cuda"`, mas
  `torch.cuda.is_available()` retornava `False` e o antigo
  `utils.get_device()` caía para a CPU **em silêncio**. Corrigido
  reinstalando do índice CUDA correspondente ao driver:
  `torch==2.13.0+cu126` / `torchvision==0.28.0+cu126`.
- **Novo `hardware.py`** — ponto único de decisão de dispositivo. Detecta
  SO, todas as placas de vídeo (sem assumir fabricante: NVIDIA, AMD, Intel
  e Apple), driver, CUDA do driver, CUDA do build, cuDNN, *compute
  capability* e suporte a bf16; escolhe o backend (CUDA/ROCm/XPU/MPS/CPU);
  configura AMP; e monitora memória e ocupação da GPU.
- **Fim do fallback silencioso.** Se a máquina tem GPU e o PyTorch não
  consegue usá-la, o projeto avisa dizendo o motivo e o comando de
  correção. Com `REQUIRE_GPU=1` (padrão), `train.py` aborta.
- **Verificação real, não declarativa.** Antes da primeira época, a GPU
  executa uma operação de verdade e o código confere que houve alocação de
  memória na placa. Durante o treino, a ocupação é amostrada continuamente
  via NVML — **97% de média** na medição feita.
- **Ganho medido: 30,9× por batch de treino** (0,436 s na GPU contra
  12,11 s na CPU, mesma máquina). As 60 épocas caem de ~49 h para ~1,9 h.
- **Precisão mista (fp16) passou a funcionar de fato.** `use_amp` já era
  `True`, mas nunca tinha efeito na build CPU-only. A **loss é calculada em
  float32, fora do `autocast`**, de propósito: `DiceLoss` soma os 262.144
  pixels da imagem e o maior valor finito em fp16 é 65.504 — em fp16 a soma
  saturaria em `+inf` e a loss viraria `NaN`. Assim o valor é o mesmo de um
  treino em precisão simples.
- **Nada de metodologia mudou**: arquitetura, dataset, máscaras, split
  70/15/15 com seed 42, loss, métricas, épocas, critérios de avaliação e
  pré-processamento seguem idênticos. O `batch_size` **continua 8** — a
  medição mostrou que ele usa só 36% da VRAM e já satura o throughput.
- **Todo o projeto usa o mesmo dispositivo**: treino, validação, teste,
  avaliação, geração de máscaras, carregamento de checkpoints, inferência
  em uma imagem e inferência em lote.

### Novidades

- **`verificar_ambiente.py`** — diagnóstico completo do ambiente, roda sem
  dependências instaladas.
- **`hardware.py`** — diagnóstico e configuração de GPU/dispositivo, com
  verificação real de que a placa está trabalhando.
- **`exemplo_medicao.py`** — demonstra a medição sobre as máscaras
  verdadeiras do dataset, sem exigir modelo treinado.
- **Contagem de imagens corrigida** nos logs de exploração: `Path.glob('*')`
  inclui arquivos ocultos (diferente do shell), então o `.gitkeep` era
  contado e os totais apareciam com uma unidade a mais.
- **Credenciais do Kaggle**: os três mecanismos aceitos passaram a ser
  documentados, incluindo o token moderno `KAGGLE_API_TOKEN` (`KGAT_...`),
  que é o formato entregue hoje pelo site. A falha de autenticação agora
  produz uma mensagem que lista as opções em vez de um *traceback* cru.
- **README** reescrito com tutorial passo a passo e solução de problemas.

### Ambiente de validação

| Componente | Versão |
| --- | --- |
| Python | 3.14.6 (Windows 10) |
| torch / torchvision | 2.13.0+cpu / 0.28.0+cpu |
| segmentation-models-pytorch | 0.5.0 |
| torchmetrics | 1.9.0 |
| albumentations | 2.0.8 |
| opencv-python-headless | 5.0.0.93 |
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| kaggle | 2.2.4 |
| pytest | 9.1.1 |

Resultado: `pytest tests/ -q` → **54 passed**.

### Ambiente de validação — GPU

| Componente | Versão |
| --- | --- |
| Python | 3.13.9 (Windows 11) |
| CPU / RAM | Intel i7-13620H (16 threads) / 16 GB |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop (sm_86) |
| Driver / CUDA do driver | 566.36 / 12.7 |
| torch / torchvision | **2.13.0+cu126 / 0.28.0+cu126** |
| cuDNN | 9.10.2 |

Resultados: `pytest tests/ -q` → **54 passed** (mesma contagem de antes da
mudança); `python hardware.py` → **Status: OK**; treino real de 2 épocas
sobre as 2.761 imagens → 97% de ocupação média da GPU, 114 s/época em
regime, loss caindo de 0,61 para 0,43.
