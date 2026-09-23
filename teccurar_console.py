"""
teccurar_console.py
====================
Painel de controle do TecCurar.

Reúne em uma janela tudo que o projeto faz — verificar o ambiente, preparar
os dados, recriar as máscaras, calibrar pelas moedas, validar, treinar,
avaliar, rodar em lote e analisar uma fotografia — sem que seja preciso
executar comandos no terminal.

    teccurar_console      janela, abas e botões
            ↓
    teccurar_tarefas      catálogo das etapas e execução em subprocesso
            ↓
    scripts do projeto    train.py, evaluate.py, calibrar_moedas.py, ...

O painel **não reimplementa nenhuma etapa**: cada botão dispara o mesmo
script que a linha de comando dispararia, com a saída aparecendo ao vivo na
janela. A aba de análise reaproveita o painel de
:mod:`teccurar_interface`.

Uso:
    python teccurar_console.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import config
from teccurar_tarefas import (
    ETAPAS,
    Artefato,
    Etapa,
    ExecutorEtapa,
    estado_projeto,
)

TITULO = "TecCurar — Painel de Controle"

COR_PRONTO = "#1f6b3a"
COR_PENDENTE = "#8a6d1f"
COR_ERRO = "#b03a2e"
COR_NEUTRA = "#444444"


class ConsoleTecCurar(tk.Tk):
    """Janela principal do painel de controle."""

    def __init__(self) -> None:
        super().__init__()
        self.title(TITULO)
        self.geometry("1180x820")
        self.minsize(1040, 720)

        self.executor = ExecutorEtapa()
        self._botoes: Dict[str, ttk.Button] = {}
        self._rotulos_estado: Dict[str, ttk.Label] = {}
        self._painel_analise = None

        self._montar()
        self.atualizar_estado()
        self.after(120, self._drenar_saida)
        self.protocol("WM_DELETE_WINDOW", self._ao_fechar)

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #

    def _montar(self) -> None:
        cabecalho = ttk.Frame(self, padding=(16, 12, 16, 6))
        cabecalho.pack(fill="x")
        ttk.Label(cabecalho, text="TecCurar",
                  font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(cabecalho, text="  painel de controle do projeto",
                  foreground="#666666").pack(side="left", pady=(7, 0))
        ttk.Button(cabecalho, text="Atualizar estado",
                   command=self.atualizar_estado).pack(side="right")

        self.abas = ttk.Notebook(self)
        self.abas.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        self.aba_etapas = ttk.Frame(self.abas)
        self.aba_analise = ttk.Frame(self.abas)
        self.abas.add(self.aba_etapas, text="  Etapas do projeto  ")
        self.abas.add(self.aba_analise, text="  Analisar ferida  ")
        self.abas.bind("<<NotebookTabChanged>>", self._ao_trocar_aba)

        self._montar_etapas()

        rodape = ttk.Frame(self, padding=(16, 0, 16, 10))
        rodape.pack(fill="x")
        self.status = ttk.Label(rodape, text="Pronto.", foreground=COR_NEUTRA)
        self.status.pack(side="left")
        self.progresso = ttk.Progressbar(rodape, mode="indeterminate", length=200)

    def _montar_etapas(self) -> None:
        """Monta a aba com o estado do projeto, as etapas e o log."""
        painel = ttk.Panedwindow(self.aba_etapas, orient="horizontal")
        painel.pack(fill="both", expand=True, padx=6, pady=6)

        esquerda = ttk.Frame(painel)
        direita = ttk.Frame(painel)
        painel.add(esquerda, weight=3)
        painel.add(direita, weight=4)

        # --- estado do projeto ------------------------------------------- #
        caixa_estado = ttk.LabelFrame(esquerda, text="Estado do projeto", padding=10)
        caixa_estado.pack(fill="x", pady=(0, 8))
        for chave, artefato in estado_projeto().items():
            linha = ttk.Frame(caixa_estado)
            linha.pack(fill="x", pady=1)
            ttk.Label(linha, text=artefato.titulo, width=24,
                      anchor="w").pack(side="left")
            rotulo = ttk.Label(linha, text="", anchor="w")
            rotulo.pack(side="left", fill="x", expand=True)
            self._rotulos_estado[chave] = rotulo

        # --- etapas -------------------------------------------------------- #
        caixa = ttk.LabelFrame(esquerda, text="Etapas", padding=(8, 6))
        caixa.pack(fill="both", expand=True)

        tela = tk.Canvas(caixa, borderwidth=0, highlightthickness=0)
        barra = ttk.Scrollbar(caixa, orient="vertical", command=tela.yview)
        interno = ttk.Frame(tela)
        interno.bind("<Configure>",
                     lambda e: tela.configure(scrollregion=tela.bbox("all")))
        janela = tela.create_window((0, 0), window=interno, anchor="nw")
        tela.bind("<Configure>", lambda e: tela.itemconfig(janela, width=e.width))
        tela.configure(yscrollcommand=barra.set)
        tela.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")

        for etapa in ETAPAS:
            bloco = ttk.Frame(interno, padding=(2, 6))
            bloco.pack(fill="x")
            topo = ttk.Frame(bloco)
            topo.pack(fill="x")
            ttk.Label(topo, text=etapa.titulo,
                      font=("Segoe UI", 10, "bold")).pack(side="left")
            botao = ttk.Button(topo, text="Executar", width=11,
                               command=lambda e=etapa: self.executar(e))
            botao.pack(side="right")
            self._botoes[etapa.chave] = botao
            ttk.Label(bloco, text=etapa.descricao, foreground="#555555",
                      wraplength=430, justify="left").pack(fill="x", pady=(2, 0))
            self._botoes[etapa.chave + "__aviso"] = ttk.Label(
                bloco, text="", foreground=COR_PENDENTE, wraplength=430,
                justify="left")
            self._botoes[etapa.chave + "__aviso"].pack(fill="x")
            ttk.Separator(interno, orient="horizontal").pack(fill="x", pady=2)

        # --- log ------------------------------------------------------------ #
        caixa_log = ttk.LabelFrame(direita, text="Saída da etapa", padding=(8, 6))
        caixa_log.pack(fill="both", expand=True)

        barra_log = ttk.Frame(caixa_log)
        barra_log.pack(fill="x", pady=(0, 6))
        self.botao_parar = ttk.Button(barra_log, text="Parar etapa",
                                      command=self.parar, state="disabled")
        self.botao_parar.pack(side="left")
        ttk.Button(barra_log, text="Limpar",
                   command=self._limpar_log).pack(side="left", padx=(6, 0))
        ttk.Button(barra_log, text="Abrir pasta de resultados",
                   command=self._abrir_resultados).pack(side="right")

        self.log = tk.Text(caixa_log, wrap="none", height=24,
                           background="#1e1e1e", foreground="#dcdcdc",
                           insertbackground="#dcdcdc", font=("Consolas", 9))
        rolagem = ttk.Scrollbar(caixa_log, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=rolagem.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        rolagem.pack(side="right", fill="y")

        self.log.tag_configure("cabecalho", foreground="#6fb3e0")
        self.log.tag_configure("ok", foreground="#7ec699")
        self.log.tag_configure("erro", foreground="#e08585")

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #

    def atualizar_estado(self) -> None:
        """Relê o disco e ajusta indicadores e botões."""
        estado = estado_projeto()
        for chave, artefato in estado.items():
            rotulo = self._rotulos_estado.get(chave)
            if rotulo is not None:
                rotulo.config(
                    text=("✓  " if artefato.pronto else "—  ") + artefato.detalhe,
                    foreground=COR_PRONTO if artefato.pronto else COR_PENDENTE)

        ocupado = self.executor.rodando
        for etapa in ETAPAS:
            motivo = etapa.indisponivel_por(estado)
            botao = self._botoes[etapa.chave]
            aviso = self._botoes[etapa.chave + "__aviso"]
            botao.config(state="disabled" if (ocupado or motivo) else "normal")
            aviso.config(text=motivo or "")

    # ------------------------------------------------------------------ #
    # Execução de etapas
    # ------------------------------------------------------------------ #

    def executar(self, etapa: Etapa) -> None:
        """Valida, confirma e dispara uma etapa."""
        if self.executor.rodando:
            messagebox.showinfo(
                "Uma etapa já está em execução",
                "Aguarde a etapa atual terminar, ou use 'Parar etapa'.")
            return

        motivo = etapa.indisponivel_por(estado_projeto())
        if motivo:
            messagebox.showwarning("Etapa indisponível", "%s\n\n%s"
                                   % (etapa.titulo, motivo))
            return

        extras = []
        if etapa.chave == "lote":
            pasta = filedialog.askdirectory(
                title="Escolha a pasta com as imagens a processar")
            if not pasta:
                return
            extras = ["--input-dir", pasta]

        if etapa.confirmar and not messagebox.askyesno(etapa.titulo, etapa.confirmar):
            return

        try:
            self.executor.iniciar(etapa, extras)
        except Exception as erro:
            messagebox.showerror(
                "Não foi possível iniciar",
                "A etapa '%s' não pôde ser iniciada.\n\nDetalhe técnico: %s"
                % (etapa.titulo, erro))
            return

        self.botao_parar.config(state="normal")
        self.progresso.pack(side="right")
        self.progresso.start(12)
        self.atualizar_estado()

    def parar(self) -> None:
        if not self.executor.rodando:
            return
        etapa = self.executor.etapa_atual
        if messagebox.askyesno(
                "Parar etapa",
                "Interromper '%s'?\n\nO que já foi gravado em disco é mantido."
                % (etapa.titulo if etapa else "etapa atual")):
            self.executor.parar()

    def _drenar_saida(self) -> None:
        """Consome a fila do executor e escreve no log."""
        try:
            while True:
                tipo, etapa, dado = self.executor.fila.get_nowait()
                if tipo == "inicio":
                    self._escrever("\n%s\n%s\n%s\n" % ("=" * 70, etapa.titulo, "=" * 70),
                                   "cabecalho")
                    self._escrever("> %s\n\n" % dado, "cabecalho")
                    self._definir_status("Executando: %s" % etapa.titulo)
                elif tipo == "saida":
                    self._escrever(dado + "\n")
                elif tipo == "fim":
                    self._finalizar(etapa, dado)
        except Exception:
            pass  # fila vazia
        self.after(120, self._drenar_saida)

    def _finalizar(self, etapa: Etapa, codigo: int) -> None:
        sucesso = codigo == 0
        self._escrever(
            "\n%s (código %d)\n" % ("Etapa concluída." if sucesso
                                    else "Etapa encerrada com erro.", codigo),
            "ok" if sucesso else "erro")
        self.progresso.stop()
        self.progresso.pack_forget()
        self.botao_parar.config(state="disabled")
        self.atualizar_estado()

        if sucesso:
            self._definir_status("%s — concluída." % etapa.titulo, COR_PRONTO)
            if etapa.produz:
                messagebox.showinfo(
                    "Etapa concluída",
                    "%s foi concluída.\n\nResultados em: %s"
                    % (etapa.titulo, etapa.produz))
        else:
            self._definir_status("%s — encerrada com erro." % etapa.titulo, COR_ERRO)
            messagebox.showerror(
                "A etapa não foi concluída",
                "%s terminou com erro (código %d).\n\n"
                "A saída completa está no painel à direita — o motivo costuma "
                "aparecer nas últimas linhas." % (etapa.titulo, codigo))

    # ------------------------------------------------------------------ #
    # Utilidades da janela
    # ------------------------------------------------------------------ #

    def _escrever(self, texto: str, tag: str = "") -> None:
        self.log.config(state="normal")
        self.log.insert("end", texto, tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _limpar_log(self) -> None:
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _abrir_resultados(self) -> None:
        """Abre a pasta de resultados no explorador de arquivos."""
        pasta = config.BASE_DIR / "Resultados"
        pasta.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(str(pasta))  # noqa: S606 - Windows
        except AttributeError:  # pragma: no cover - outros sistemas
            import subprocess
            subprocess.Popen(["xdg-open", str(pasta)])
        except Exception as erro:
            messagebox.showerror("Não foi possível abrir a pasta",
                                 "Abra manualmente:\n%s\n\nDetalhe: %s" % (pasta, erro))

    def _ao_trocar_aba(self, _evento) -> None:
        """Monta a aba de análise só quando ela é aberta pela primeira vez.

        Construí-la na abertura da janela custaria o carregamento do modelo
        antes de o usuário pedir por ele.
        """
        if self.abas.index("current") != 1 or self._painel_analise is not None:
            return
        try:
            from teccurar_interface import PainelAnalise
            self._painel_analise = PainelAnalise(self.aba_analise, com_cabecalho=False)
            self._painel_analise.pack(fill="both", expand=True)
        except Exception as erro:
            ttk.Label(self.aba_analise, padding=24, foreground=COR_ERRO,
                      wraplength=700, justify="left",
                      text="A tela de análise não pôde ser aberta.\n\n"
                           "Detalhe técnico: %s" % erro).pack()

    def _definir_status(self, texto: str, cor: str = COR_NEUTRA) -> None:
        self.status.config(text=texto, foreground=cor)

    def _ao_fechar(self) -> None:
        if self.executor.rodando:
            etapa = self.executor.etapa_atual
            if not messagebox.askyesno(
                    "Etapa em execução",
                    "'%s' ainda está rodando.\n\nFechar o painel vai interrompê-la. "
                    "Deseja fechar mesmo assim?"
                    % (etapa.titulo if etapa else "Uma etapa")):
                return
            self.executor.parar()
        self.destroy()


def main() -> int:
    """Abre o painel de controle do TecCurar."""
    try:
        from PIL import Image  # noqa: F401  (a aba de análise depende dele)
    except ImportError:
        print("ERRO: o painel precisa do Pillow. Instale com: pip install pillow")
        return 1
    ConsoleTecCurar().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
