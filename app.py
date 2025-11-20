# ==========================================================
# SHINY DASHBOARD DE PREVISÕES MACROECONÔMICAS
# Objetivo: montar uma aplicação Shiny em Python que exibe gráficos
# e tabelas das previsões geradas (Câmbio, IPCA, PIB, Selic),
# permitindo ao usuário filtrar modelos, iniciar janela temporal e
# mostrar/ocultar intervalos de confiança.
# ==========================================================


# Bibliotecas ----
from shiny import App, ui, render                 # framework Shiny para Python
from faicons import icon_svg                      # ícones SVG para a UI
from shinyswatch import theme                     # temas prontos para o app
import pandas as pd
import plotnine as p9                             # gramática de gráficos (ggplot-like)
from mizani import breaks                         # breaks para eixos y


# Dados ----
# Carrega previsões previamente salvas em parquet
cambio = pd.read_parquet("previsao/cambio.parquet")
ipca = pd.read_parquet("previsao/ipca.parquet")
pib = pd.read_parquet("previsao/pib.parquet")
selic = pd.read_parquet("previsao/selic.parquet")

# Dicionário com limites e valor inicial do input de data no sidebar
datas = {
    "min": pib.index.min().date(),        # data mínima disponível (para o input)
    "max": selic.index.max().date(),      # data máxima disponível (para o input)
    "value": pib.index[-36].date()        # data inicial padrão (ex.: 36 observações atrás)
}

# Lista de modelos únicos (exclui rótulos de séries como 'IPCA', 'PIB', etc.)
modelos = (
    pd.concat([
        cambio,
        ipca,
        pib,
        selic
    ])
    .query("Tipo not in ['Câmbio', 'IPCA', 'PIB', 'Selic']")
    .Tipo
    .unique()
    .tolist()
)


# Interface do Usuário ----
app_ui = ui.page_navbar(
    # Conteúdo principal: painéis com gráficos e tabelas para cada indicador
    ui.nav_panel(
        "",
        ui.layout_columns(
            ui.navset_card_underline(
                ui.nav_panel("", ui.output_plot("ipca_plt"), icon = icon_svg("chart-line"), value = "plt"),
                ui.nav_panel("", ui.output_data_frame("ipca_tbl"), icon = icon_svg("table"), value = "tbl"),
                title = "Inflação (IPCA)",
                selected = "plt"   # seleciona o gráfico por padrão
            ),
            ui.navset_card_underline(
                ui.nav_panel("", ui.output_plot("cambio_plt"), icon = icon_svg("chart-line"), value = "plt"),
                ui.nav_panel("", ui.output_data_frame("cambio_tbl"), icon = icon_svg("table"), value = "tbl"),
                title = "Taxa de Câmbio (BRL/USD)",
                selected = "plt"
            )
        ),
        ui.layout_columns(
            ui.navset_card_underline(
                ui.nav_panel("", ui.output_plot("pib_plt"), icon = icon_svg("chart-line"), value = "plt"),
                ui.nav_panel("", ui.output_data_frame("pib_tbl"), icon = icon_svg("table"), value = "tbl"),
                title = "Atividade Econômica (PIB)",
                selected = "plt"
            ),
            ui.navset_card_underline(
                ui.nav_panel("", ui.output_plot("selic_plt"), icon = icon_svg("chart-line"), value = "plt"),
                ui.nav_panel("", ui.output_data_frame("selic_tbl"), icon = icon_svg("table"), value = "tbl"),
                title = "Taxa de Juros (SELIC)",
                selected = "plt"
            )
        )
    ),
    # Cabeçalho com logo
    title = ui.img(
        src = "https://aluno.analisemacro.com.br/download/59787/?tmstv=1712933415",
        height = 35
    ),
    window_title = "Painel de Previsões",
    fillable = True,
    fillable_mobile = True,
    theme = theme.minty,   # tema visual
    # Sidebar com inputs e descrição
    sidebar = ui.sidebar(
        ui.markdown("Acompanhe as previsões automatizadas dos principais indicadores macroeconômicos do Brasil e simule cenários alternativos em um mesmo dashboard."),
        # Input: seleção dos modelos a exibir (multi-select)
        ui.input_selectize(
            id = "modelo",
            label = ui.strong("Selecionar modelos:"),
            choices = modelos,
            selected = modelos,
            multiple = True,
            width = "100%",
            options = {"plugins": ["clear_button"]}
        ),
        # Input: data inicial do gráfico (date picker)
        ui.input_date(
            id = "inicio",
            label = ui.strong("Início do gráfico:"),
            value = datas["value"],
            min = datas["min"],
            max = datas["max"],
            format = "mm/yyyy",
            startview = "year",
            language = "pt-BR",
            width = "100%"
        ),
        # Input: checkbox para exibir/ocultar intervalo de confiança
        ui.input_checkbox(
            id = "ic",
            label = ui.strong("Intervalo de confiança"),
            value = True,
            width = "100%"
        ),
        ui.markdown("Elaboração: Análise Macro")
    )
)


# Servidor ----
def server(input, output, session):
    """
    Funções principais:
    - plotar_grafico: constrói gráfico ggplot-like com linhas e ribbons (IC opcional)
    - imprimir_tabela: formata DataFrame para exibição em grid
    - bindings de render (ipca_plt, cambio_plt, pib_plt, selic_plt, e as tabelas)
    """

    def plotar_grafico(y, df, y_label):
        """
        Recebe:
        - y: nome da série referência (ex.: "IPCA")
        - df: DataFrame com previsões (index datetime, col 'Valor', 'Intervalo Inferior/Superior', 'Tipo')
        - y_label: rótulo do eixo y

        Retorna um objeto plotnine (ggplot) pronto para renderizar na UI.
        """

        modelos1 = [y] + list(input.modelo())                      # série base + modelos selecionados
        modelos2 = df.query("Tipo != @y").Tipo.unique().tolist()   # todos os modelos disponíveis exceto a série y

        data = input.inicio()                                      # data inicial escolhida pelo usuário

        # Filtra e ordena os dados a partir da data selecionada,
        # define a ordem categórica para garantir cores consistentes
        df_tmp = (
            df
            .query("index >= @data")
            .reset_index()
            .assign(Tipo = lambda x: pd.Categorical(x.Tipo, [y] + modelos2))
            .query("Tipo in @modelos1")
        )

        # Função interna que retorna geom_ribbon para intervalo de confiança
        def plotar_ic():
            ic = p9.geom_ribbon(
                mapping = p9.aes(ymin = "Intervalo Inferior", ymax = "Intervalo Superior", fill = "Tipo"),
                show_legend = False,
                color = "none",
                alpha = 0.25
            )
            # exibe o ribbon apenas se input.ic() for True
            if input.ic():
                return ic
            else:
                return None

        # Monta o gráfico com linhas, possível ribbon, e escalas/legenda customizadas
        plt = (
            p9.ggplot(df_tmp) +
            p9.aes(x = "index", y = "Valor", color = "Tipo") +
            plotar_ic() +
            p9.geom_line() +
            p9.scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
            p9.scale_y_continuous(breaks = breaks.breaks_extended(n = 6)) +
            p9.scale_color_manual(
                values = {
                    "IPCA": "black",
                    "Câmbio": "black",
                    "PIB": "black",
                    "Selic": "black",
                    "IA": "green",
                    "Ridge": "blue",
                    "Bayesian Ridge": "orange",
                    "Huber": "red",
                    "Ensemble": "brown"
                }
            ) +
            p9.scale_fill_manual(
                values = {
                    "IA": "green",
                    "Ridge": "blue",
                    "Bayesian Ridge": "orange",
                    "Huber": "red",
                    "Ensemble": "brown"
                }
            ) +
            p9.labs(
                y = y_label,
                x = "",
                color = ""
            ) +
            p9.theme(legend_position = "bottom")
        )

        return plt

    def imprimir_tabela(df, y):
        """
        Formata um DataFrame para exibição em grid:
        - reseta index para coluna 'Data'
        - renomeia colunas
        - remove linhas onde 'Modelo' == y (a série de referência)
        - formata data como mm/YYYY e arredonda valores
        - retorna um render.DataGrid compatível com Shiny
        """
        df = (
            df
            .reset_index()
            .rename(columns = {"index": "Data", "Valor": "Previsão", "Tipo": "Modelo"})
            .query("Modelo != @y")
            .assign(Data = lambda x: x.Data.dt.strftime("%m/%Y"))
            .round(2)
        )
        return render.DataGrid(df, summary = False)

    # Bindings dos gráficos: cada função renderiza o plot correspondente
    @render.plot
    def ipca_plt():
        return plotar_grafico(y = "IPCA", df = ipca, y_label = "Var. %")

    @render.plot
    def cambio_plt():
        return plotar_grafico(y = "Câmbio", df = cambio, y_label = "R\\$/US\\$")

    @render.plot
    def pib_plt():
        return plotar_grafico(y = "PIB", df = pib, y_label = "Var. % anual")

    @render.plot
    def selic_plt():
        return plotar_grafico(y = "Selic", df = selic, y_label = "% a.a.")

    # Bindings das tabelas: retornam grids formatadas
    @render.data_frame
    def ipca_tbl():
        return imprimir_tabela(ipca, "IPCA")

    @render.data_frame
    def cambio_tbl():
        return imprimir_tabela(cambio, "Câmbio")

    @render.data_frame
    def pib_tbl():
        return imprimir_tabela(pib, "PIB")

    @render.data_frame
    def selic_tbl():
        return imprimir_tabela(selic, "Selic")


# Shiny dashboard ----
# Instancia a aplicação com UI e servidor definidos acima
app = App(app_ui, server)
