"""
teccurar_interface.py
======================
Interface gráfica do TecCurar.

Permite escolher a fotografia de uma ferida, ver a segmentação sobreposta à
imagem, ler as medidas e salvar um relatório em PDF — sem executar código,
editar parâmetros ou treinar nada.

    interface  ->  teccurar_pipeline  ->  predict / measure / hardware
               ->  teccurar_relatorio ->  PDF

Este arquivo contém apenas apresentação e tratamento de erro. Toda a lógica
de modelo, medição e calibração vive em :mod:`teccurar_pipeline`, e a
geração do PDF em :mod:`teccurar_relatorio`.

Construída com ``tkinter``, da biblioteca padrão: não acrescenta nenhuma
dependência ao projeto e abre como aplicativo de desktop, que é o formato
adequado a um projeto local de linha de comando como este.

Uso:
    python teccurar_interface.py
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

import config
from teccurar_pipeline import (
    CALIBRACAO_AUSENTE,
    ErroTecCurar,
    ResultadoAnalise,
    analisar_imagem,
    linhas_de_medidas,
    localizar_checkpoint,
    montar_escala,
)
from teccurar_relatorio import gerar_relatorio, nome_padrao

TITULO = "TecCurar — Análise de Feridas"
LARGURA_VISTA = 320


class PainelAnalise(ttk.Frame):
    """Painel de análise de uma fotografia.

    É um ``Frame`` e não uma janela para que sirva tanto à janela autônoma
    (:class:`AplicacaoTecCurar`) quanto a uma aba do painel de controle em
    :mod:`teccurar_console`, sem duplicar a tela.
    """

    def __init__(self, mestre, com_cabecalho: bool = True) -> None:
        super().__init__(mestre)

        self._com_cabecalho = com_cabecalho
        self._caminho: Optional[Path] = None
        self._resultado: Optional[ResultadoAnalise] = None
        self._miniaturas: list = []   # evita coleta das imagens pelo Tk
        self._sem_modelo = False      # nenhum checkpoint na ultima verificacao

        self._montar()
        self._verificar_modelo()

    # ------------------------------------------------------------------ #
    # Construção da tela
    # ------------------------------------------------------------------ #

    def _montar(self) -> None:
        # Dentro do painel de controle o título já aparece na janela; repeti-lo
        # na aba só roubaria altura das imagens.
        if self._com_cabecalho:
            cabecalho = ttk.Frame(self, padding=(14, 10, 14, 4))
            cabecalho.pack(fill="x")
            ttk.Label(cabecalho, text="TecCurar",
                      font=("Segoe UI", 17, "bold")).pack(side="left")
            ttk.Label(cabecalho, text="  segmentação e medição de feridas",
                      foreground="#666666").pack(side="left", pady=(6, 0))

        barra = ttk.Frame(self, padding=(14, 4))
        barra.pack(fill="x")
        self.botao_abrir = ttk.Button(barra, text="Selecionar imagem da ferida",
                                      command=self._escolher_imagem)
        self.botao_abrir.pack(side="left")
        self.botao_relatorio = ttk.Button(barra, text="Gerar relatório",
                                          command=self._gerar_relatorio, state="disabled")
        self.botao_relatorio.pack(side="left", padx=(8, 0))
        # Só aparece quando falta modelo: treinar é demorado e é comum deixar
        # esta tela aberta enquanto o treinamento roda.
        self.botao_reverificar = ttk.Button(barra, text="Procurar modelo novamente",
                                            command=self._reverificar_modelo)
        self.rotulo_arquivo = ttk.Label(barra, text="Nenhuma imagem selecionada",
                                        foreground="#666666")
        self.rotulo_arquivo.pack(side="left", padx=(14, 0))

        self._montar_escala()

        # --- três vistas ------------------------------------------------- #
        vistas = ttk.Frame(self, padding=(14, 8))
        vistas.pack(fill="both", expand=True)
        self.vistas = {}
        for i, (chave, titulo) in enumerate((
            ("original", "Fotografia original"),
            ("mascara", "Máscara segmentada"),
            ("sobreposicao", "Fotografia + máscara"),
        )):
            quadro = ttk.LabelFrame(vistas, text=titulo, padding=6)
            quadro.grid(row=0, column=i, padx=6, sticky="nsew")
            vistas.columnconfigure(i, weight=1)
            rotulo = ttk.Label(quadro, anchor="center")
            rotulo.pack(fill="both", expand=True)
            self.vistas[chave] = rotulo

        # --- medidas ----------------------------------------------------- #
        medidas = ttk.LabelFrame(self, text="Medidas da ferida", padding=8)
        medidas.pack(fill="x", padx=20, pady=(4, 6))
        self.tabela = ttk.Treeview(medidas, columns=("px", "cm"), show="tree headings", height=4)
        self.tabela.heading("#0", text="Medida")
        self.tabela.heading("px", text="Pixels")
        self.tabela.heading("cm", text="Centímetros")
        self.tabela.column("#0", width=220, anchor="w")
        self.tabela.column("px", width=200, anchor="center")
        self.tabela.column("cm", width=200, anchor="center")
        self.tabela.pack(fill="x")

        self.rotulo_calibracao = ttk.Label(medidas, text="", foreground="#8a6d1f",
                                           wraplength=980, justify="left")
        self.rotulo_calibracao.pack(fill="x", pady=(6, 0))

        rodape = ttk.Frame(self, padding=(14, 4, 14, 10))
        rodape.pack(fill="x")
        self.status = ttk.Label(rodape, text="Pronto.", foreground="#444444")
        self.status.pack(side="left")
        self.progresso = ttk.Progressbar(rodape, mode="indeterminate", length=180)

    def _montar_escala(self) -> None:
        """Campos opcionais de referência física."""
        caixa = ttk.LabelFrame(
            self, padding=8,
            text="Escala física (opcional — sem isso as medidas saem apenas em pixels)")
        caixa.pack(fill="x", padx=14, pady=(6, 2))

        self.modo_escala = tk.StringVar(value="nenhuma")
        ttk.Radiobutton(caixa, text="Sem referência", value="nenhuma",
                        variable=self.modo_escala, command=self._alternar_escala
                        ).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Radiobutton(caixa, text="Informar pixels/cm", value="fator",
                        variable=self.modo_escala, command=self._alternar_escala
                        ).grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Radiobutton(caixa, text="Objeto de referência na foto", value="objeto",
                        variable=self.modo_escala, command=self._alternar_escala
                        ).grid(row=0, column=2, sticky="w")

        self.campo_fator = ttk.Entry(caixa, width=10)
        self.rot_fator = ttk.Label(caixa, text="pixels por cm:")
        self.campo_ref_px = ttk.Entry(caixa, width=10)
        self.rot_ref_px = ttk.Label(caixa, text="objeto (pixels):")
        self.campo_ref_cm = ttk.Entry(caixa, width=10)
        self.rot_ref_cm = ttk.Label(caixa, text="objeto (cm):")

        self.rot_fator.grid(row=1, column=0, sticky="e", pady=(8, 0))
        self.campo_fator.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.rot_ref_px.grid(row=2, column=0, sticky="e", pady=(4, 0))
        self.campo_ref_px.grid(row=2, column=1, sticky="w", pady=(4, 0))
        self.rot_ref_cm.grid(row=2, column=2, sticky="e", pady=(4, 0))
        self.campo_ref_cm.grid(row=2, column=3, sticky="w", pady=(4, 0))
        self._alternar_escala()

    def _alternar_escala(self) -> None:
        """Mostra apenas os campos do modo escolhido."""
        modo = self.modo_escala.get()
        for widget in (self.rot_fator, self.campo_fator):
            widget.grid_remove() if modo != "fator" else widget.grid()
        for widget in (self.rot_ref_px, self.campo_ref_px, self.rot_ref_cm, self.campo_ref_cm):
            widget.grid_remove() if modo != "objeto" else widget.grid()

    # ------------------------------------------------------------------ #
    # Estado do modelo
    # ------------------------------------------------------------------ #

    def _verificar_modelo(self, avisar: bool = True) -> bool:
        """Confere se há modelo treinado e ajusta a tela ao que encontrar.

        A ausência de checkpoint nunca desabilita a tela em definitivo, só
        até a próxima verificação: o modelo costuma aparecer depois que esta
        tela foi aberta — treinar leva minutos, e o painel de controle roda o
        treinamento com a análise já montada ao lado.

        Args:
            avisar: mostra o diálogo de aviso quando não há modelo. Fica em
                ``False`` nas reconferências automáticas, que não devem
                interromper o usuário.

        Returns:
            ``True`` se há um modelo disponível.
        """
        try:
            caminho = localizar_checkpoint()
        except ErroTecCurar as erro:
            self._sem_modelo = True
            self.botao_abrir.config(state="disabled")
            self.botao_reverificar.pack(side="left", padx=(8, 0))
            self._definir_status("Nenhum modelo treinado disponível.", erro=True)
            if avisar:
                messagebox.showwarning("Modelo não disponível", str(erro))
            return False

        self._sem_modelo = False
        self.botao_abrir.config(state="normal")
        self.botao_reverificar.pack_forget()
        self._definir_status("Modelo disponível: %s" % caminho.name)
        return True

    def _reverificar_modelo(self) -> None:
        """Procura o modelo de novo, a pedido do usuário."""
        if self._verificar_modelo(avisar=False):
            messagebox.showinfo(
                "Modelo encontrado",
                "Um modelo treinado foi encontrado. A análise de fotografias "
                "já está disponível.")
        else:
            messagebox.showwarning(
                "Modelo ainda não disponível",
                "Continua não havendo nenhum modelo treinado em 'checkpoints'.\n\n"
                "Conclua o treinamento — etapa 'Treinar o modelo' no painel de "
                "controle, ou 'python train.py' — e procure novamente.")

    def revalidar_modelo(self) -> None:
        """Reconfere a disponibilidade do modelo, sem diálogos.

        Chamada pelo painel de controle a cada volta à aba de análise. Só age
        quando falta modelo, para não apagar da tela o resultado de uma
        análise já concluída.
        """
        if self._sem_modelo:
            self._verificar_modelo(avisar=False)

    # ------------------------------------------------------------------ #
    # Ações
    # ------------------------------------------------------------------ #

    def _escolher_imagem(self) -> None:
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem da ferida",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png"), ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return                      # o usuário cancelou: não é erro
        self._caminho = Path(caminho)
        self.rotulo_arquivo.config(text=self._caminho.name)
        self._analisar()

    def _ler_escala(self):
        """Lê os campos de escala, recusando valores inválidos."""
        modo = self.modo_escala.get()
        try:
            if modo == "fator":
                valor = float(self.campo_fator.get().replace(",", "."))
                if valor <= 0:
                    raise ValueError
                return montar_escala(pixels_por_cm=valor)
            if modo == "objeto":
                px = float(self.campo_ref_px.get().replace(",", "."))
                cm = float(self.campo_ref_cm.get().replace(",", "."))
                if px <= 0 or cm <= 0:
                    raise ValueError
                return montar_escala(referencia_pixels=px, referencia_cm=cm)
        except ValueError:
            messagebox.showwarning(
                "Escala inválida",
                "Os valores de escala precisam ser números maiores que zero.\n\n"
                "A análise vai continuar apenas em pixels.")
            self.modo_escala.set("nenhuma")
            self._alternar_escala()
        return None, CALIBRACAO_AUSENTE

    def _analisar(self) -> None:
        """Dispara a análise em uma thread, mantendo a janela responsiva."""
        escala, status = self._ler_escala()
        self._ocupado(True, "Analisando a fotografia...")

        def tarefa():
            try:
                resultado = analisar_imagem(self._caminho, escala=escala,
                                            status_calibracao=status)
                self.after(0, self._mostrar_resultado, resultado)
            except ErroTecCurar as erro:
                self.after(0, self._mostrar_erro, "Não foi possível analisar", str(erro))
            except Exception as erro:  # rede de segurança: nada escapa cru
                self.after(0, self._mostrar_erro, "Erro inesperado",
                           "Ocorreu um erro inesperado ao processar a imagem.\n\n"
                           "Detalhe técnico: %s" % erro)

        threading.Thread(target=tarefa, daemon=True).start()

    def _gerar_relatorio(self) -> None:
        if self._resultado is None:
            messagebox.showinfo("Nada para relatar",
                                "Analise uma fotografia antes de gerar o relatório.")
            return
        destino = filedialog.asksaveasfilename(
            title="Salvar relatório", defaultextension=".pdf",
            initialfile=nome_padrao(self._resultado.quando),
            filetypes=[("PDF", "*.pdf")])
        if not destino:
            return
        try:
            caminho = gerar_relatorio(self._resultado, Path(destino))
        except Exception as erro:
            self._mostrar_erro("Falha ao gerar o relatório",
                               "O relatório não pôde ser salvo.\n\n"
                               "Verifique se o arquivo não está aberto em outro "
                               "programa e se há permissão de escrita na pasta.\n\n"
                               "Detalhe técnico: %s" % erro)
            return
        self._definir_status("Relatório salvo em %s" % caminho.name)
        messagebox.showinfo("Relatório gerado", "Relatório salvo em:\n%s" % caminho)

    # ------------------------------------------------------------------ #
    # Apresentação
    # ------------------------------------------------------------------ #

    def _mostrar_resultado(self, resultado: ResultadoAnalise) -> None:
        self._resultado = resultado
        self._ocupado(False)

        self._pintar("original", resultado.imagem_rgb)
        self._pintar("mascara", np.dstack([resultado.mascara * 255] * 3))
        self._pintar("sobreposicao", resultado.sobreposicao)

        self.tabela.delete(*self.tabela.get_children())
        for nome, px, cm in linhas_de_medidas(resultado):
            self.tabela.insert("", "end", text=nome, values=(px, cm))

        if resultado.tem_centimetros:
            self.rotulo_calibracao.config(
                text="Conversão física aplicada — %.2f pixels/cm (%s)."
                     % (resultado.escala.pixels_per_cm, resultado.status_calibracao),
                foreground="#1f6b3a")
        else:
            self.rotulo_calibracao.config(
                text=("Conversão para centímetros não disponível: nenhuma referência "
                      "física foi informada para esta fotografia. As medidas acima "
                      "estão em pixels. Para obter centímetros, informe o fator "
                      "pixels/cm ou fotografe um objeto de tamanho conhecido junto "
                      "à ferida e use a opção de objeto de referência."),
                foreground="#8a6d1f")

        self.botao_relatorio.config(state="normal")
        plural = "s" if len(resultado.feridas) > 1 else ""
        self._definir_status("%d ferida%s detectada%s em %s."
                             % (len(resultado.feridas), plural, plural,
                                resultado.caminho_imagem.name))

    def _pintar(self, chave: str, arte: np.ndarray) -> None:
        """Mostra um array RGB no rótulo correspondente."""
        from PIL import Image, ImageTk

        altura, largura = arte.shape[:2]
        escala = LARGURA_VISTA / max(largura, 1)
        nova = (max(1, int(largura * escala)), max(1, int(altura * escala)))
        imagem = Image.fromarray(arte.astype(np.uint8)).resize(nova, Image.LANCZOS)
        foto = ImageTk.PhotoImage(imagem)
        self.vistas[chave].config(image=foto)
        self._miniaturas.append(foto)
        if len(self._miniaturas) > 12:
            self._miniaturas = self._miniaturas[-6:]

    def _mostrar_erro(self, titulo: str, mensagem: str) -> None:
        self._ocupado(False)
        self._definir_status(titulo + ".", erro=True)
        messagebox.showerror(titulo, mensagem)

    def _ocupado(self, ocupado: bool, mensagem: str = "") -> None:
        if ocupado:
            self.botao_abrir.config(state="disabled")
            self.botao_relatorio.config(state="disabled")
            self.progresso.pack(side="right")
            self.progresso.start(12)
            self._definir_status(mensagem)
        else:
            self.progresso.stop()
            self.progresso.pack_forget()
            if not self._sem_modelo:
                self.botao_abrir.config(state="normal")

    def _definir_status(self, texto: str, erro: bool = False) -> None:
        self.status.config(text=texto, foreground="#b03a2e" if erro else "#444444")


class AplicacaoTecCurar(tk.Tk):
    """Janela autônoma com apenas a análise de fotografias.

    Para gerenciar o projeto inteiro — preparar dados, treinar, avaliar —
    use :mod:`teccurar_console`, que traz esta mesma tela como uma aba.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title(TITULO)
        self.geometry("1080x760")
        self.minsize(960, 680)
        PainelAnalise(self).pack(fill="both", expand=True)


def main() -> int:
    """Abre a janela de análise do TecCurar."""
    try:
        from PIL import Image  # noqa: F401  (usado em _pintar)
    except ImportError:
        print("ERRO: a interface precisa do Pillow. Instale com: pip install pillow")
        return 1
    AplicacaoTecCurar().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
