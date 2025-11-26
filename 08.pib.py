# ==========================================================
# MÓDULO DE PREPARAÇÃO, MODELAGEM E PREVISÃO DO PIB
# Objetivo: carregar dados, transformar séries conforme metadados,
# montar cenários para variáveis exógenas, estimar modelos de previsão
# e gerar previsões (Ridge, Bayesian Ridge e IA Gemini).
# ==========================================================


# =========== Bibliotecas
from skforecast.ForecasterAutoreg import ForecasterAutoreg  # Forecaster autoregressivo com exógenas
from sklearn.linear_model import BayesianRidge, HuberRegressor  # Regressors usados nos modelos
from sklearn.preprocessing import PowerTransformer  # Transformação para estabilizar variância
from io import StringIO  # Ler string como arquivo (usado para converter resposta IA)
import google.generativeai as genai  # Cliente para chamada ao modelo generativo (Gemini)
import pandas as pd
import numpy as np
import os

# =========== Bibliotecas adicionais necessárias
import time
from datetime import datetime, timedelta


# =========== Definições e configurações globais
h = 4                                           # horizonte de previsão (4 trimestres)
inicio_treino = pd.to_datetime("1997-10-01")    # início da amostra usada nos modelos
semente = 1984                                  # semente para reprodutibilidade


# =========== Função para transformar dados, conforme definido nos metadados
def transformar(x, tipo):
    """
    Aplica a transformação especificada nos metadados.
    Diferenciações, log, combinações etc.
    """
    switch = {
        "1": lambda x: x,
        "2": lambda x: x.diff(),
        "3": lambda x: x.diff().diff(),
        "4": lambda x: np.log(x),
        "5": lambda x: np.log(x).diff(),
        "6": lambda x: np.log(x).diff().diff()
    }

    if tipo not in switch:
        raise ValueError("Tipo inválido")

    return switch[tipo](x)


# =========== Planilha de metadados
# Lê do Google Sheets quais transformações aplicar em cada variável
metadados = (
    pd.read_excel(
        io="https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name="Metadados",
        dtype=str,
        index_col="Identificador"
    )
    .filter(["Transformação"])
)


# =========== Importa dados online
# Dados mensais e trimestrais previamente tratados
dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")
dados_brutos_t = pd.read_parquet("dados/df_trimestral.parquet")


# =========== Converte frequência
# Converte as séries para frequência trimestral (QS) e junta tudo
dados_tratados = (
    dados_brutos_m
    .resample("QS")   # converte para trimestre por média
    .mean()
    .join(
        other=(
            dados_brutos_t
            .set_index(pd.PeriodIndex(dados_brutos_t.index, freq="Q").to_timestamp())  # ajusta datas trimestrais
            .resample("QS")
            .mean()
        ),
        how="outer"
    )
    .rename_axis("data", axis="index")
)


# =========== Separa Y
# Variável alvo = PIB trimestral
y = dados_tratados.pib.dropna()


# =========== Separa X
# Exógenas = todas menos o PIB
x = dados_tratados.drop(labels=["pib"], axis="columns").copy()


# =========== Computa transformações
# Aplica a transformação especificada no metadado de cada variável
for col in x.drop(labels=["saldo_caged_antigo", "saldo_caged_novo"], axis="columns").columns.to_list():
    x[col] = transformar(x[col], metadados.loc[col, "Transformação"])


# =========== Filtra amostra
# Mantém apenas período >= início do treino
y = y[y.index >= inicio_treino]

# Dados exógenos além da última observação de y (para previsão futura)
x_alem_de_y = x.query("index >= @y.index.max()")

# Dados exógenos alinhados ao período de treino
x = x.query("index >= @inicio_treino and index <= @y.index.max()")


# =========== Conta proporção de NAs por coluna
prop_na = x.isnull().sum() / y.shape[0]


# =========== Remove variáveis com mais de 20% de NAs
x = x.drop(labels=prop_na[prop_na >= 0.2].index.to_list(), axis="columns")


# =========== Preenche NAs restantes
# Estratégia: backward fill + forward fill
x = x.bfill().ffill()


# =========== Seleção final de variáveis (definida manualmente)
x_reg = [
    "uci_ind_fgv",
    "expec_pib",
    "prod_ind_metalurgia"
]
# =========== + 2 lags (definidos no ForecasterAutoreg)


# =========== Reestima os 2 melhores modelos com amostra completa
# Modelo 1: Ridge
modelo1 = ForecasterAutoreg(
    regressor=Ridge(),
    lags=2,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo1.fit(y, x[x_reg])

# Modelo 2: Bayesian Ridge
modelo2 = ForecasterAutoreg(
    regressor=BayesianRidge(),
    lags=2,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo2.fit(y, x[x_reg])


# =========== Período de previsão fora da amostra
# Datas trimestrais futuras que o modelo deve prever
periodo_previsao = pd.date_range(
    start=modelo1.last_window.index[1] + pd.offsets.QuarterBegin(1),
    end=modelo1.last_window.index[1] + pd.offsets.QuarterBegin(h + 1),
    freq="QS"
)


# =========== Cenário da UCI (uci_ind_fgv)
# Calcula mediana por trimestre e projeta valores futuros
dados_cenario_uci_ind_fgv = (
    x
    .filter(["uci_ind_fgv"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(trim=lambda x: x.index.quarter)
    .groupby(["trim"], as_index=False).uci_ind_fgv.median()
    .set_index("trim")
    .join(
        other=(
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(trim=lambda x: x.data.dt.quarter)
            .drop("data", axis="columns")
            .reset_index()
            .set_index("trim")
        ),
        how="outer"
    )
    .sort_values(by="data")
    .set_index("data")
)


# =========== Coleta dados do Focus (Expectativas PIB)
dados_focus_expec_pib = pd.read_csv(
    filepath_or_buffer=f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTrimestrais?$filter=Indicador%20eq%20'PIB%20Total'%20and%20baseCalculo%20eq%200%20and%20Data%20ge%20'{periodo_previsao.min().strftime('%Y-%m-%d')}'&$format=text/csv",
    decimal=",",
    converters={"Data": pd.to_datetime}
)


# =========== Identifica a data do relatório Focus usada para o cenário
# Seleciona a data mais recente com previsões completas
data_focus_expec_pib = (
    dados_focus_expec_pib
    .assign(
        DataReferencia=lambda x: pd.PeriodIndex(
            x.DataReferencia.str.replace(r"(\d{1})/(\d{4})", r"\2-Q\1", regex=True),
            freq="Q"
        ).to_timestamp()
    )
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[1]")
    .Data.value_counts()
    .to_frame()
    .reset_index()
    .query("count >= @h")
    .query("Data == Data.max()")
    .head(1).Data.to_list()[0]
)


# =========== Constrói cenário para Expectativas PIB
dados_cenario_expec_pib = (
    dados_focus_expec_pib
    .assign(
        DataReferencia=lambda x: pd.PeriodIndex(
            x.DataReferencia.str.replace(r"(\d{1})/(\d{4})", r"\2-Q\1", regex=True),
            freq="Q"
        ).to_timestamp()
    )
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[1]")
    .query("Data == @data_focus_expec_pib")
    .query("DataReferencia in @periodo_previsao")
    .sort_values(by="DataReferencia")
    .set_index("DataReferencia")
    .filter(["Mediana"])
    .rename(columns={"Mediana": "expec_pib"})
    .dropna()
)


# =========== Constrói cenário para Produção Industrial (metalurgia)
dados_cenario_prod_ind_metalurgia = (
    x
    .filter(["prod_ind_metalurgia"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(trim=lambda x: x.index.quarter)
    .groupby(["trim"], as_index=False).prod_ind_metalurgia.median()
    .set_index("trim")
    .join(
        other=(
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(trim=lambda x: x.data.dt.quarter)
            .drop("data", axis="columns")
            .reset_index()
            .set_index("trim")
        ),
        how="outer"
    )
    .sort_values(by="data")
    .set_index("data")
)


# =========== Junta todos os cenários
dados_cenarios = (
    dados_cenario_uci_ind_fgv
    .join(
        other=[
            dados_cenario_expec_pib,
            dados_cenario_prod_ind_metalurgia
        ],
        how="outer"
    )
    .asfreq("QS")   # garante frequência trimestral regular
)


# =========== Produz previsões dos modelos tradicionais
previsao1 = (
    modelo1.predict_interval(
        steps=h,
        exog=dados_cenarios,
        n_boot=5000,
        random_state=semente
    )
    .assign(Tipo="Ridge")
    .rename(
        columns={
            "pred": "Valor",
            "lower_bound": "Intervalo Inferior",
            "upper_bound": "Intervalo Superior"
        }
    )
)

previsao2 = (
    modelo2.predict_interval(
        steps=h,
        exog=dados_cenarios,
        n_boot=5000,
        random_state=semente
    )
    .assign(Tipo="Bayesian Ridge")
    .rename(
        columns={
            "pred": "Valor",
            "lower_bound": "Intervalo Inferior",
            "upper_bound": "Intervalo Superior"
        }
    )
)


# =========== Salva CSV de entrada para IA
y.to_frame().join(x[x_reg]).to_csv("dados/pib.csv")

# Prompt enviado ao Gemini
prompt = f"""
Considere que a data atual é {pd.to_datetime("today").strftime("%B %d, %Y")}. 
Apresente sua melhor previsão do Produto Interno Bruto (PIB) do Brasil, medida em variação percentual 
anual (taxa acumulada em quatro trimestres em relação ao mesmo período do ano anterior) e divulgada pelo 
IBGE, para {periodo_previsao.min().to_period(freq = "Q").strftime("%Y Q%q")} a {periodo_previsao.max().to_period(freq = "Q").strftime("%Y Q%q")}. 
Utilize os dados históricos contidos no arquivo CSV anexo chamado "pib.csv", onde a coluna "pib" é a variável alvo, "data" é a 
coluna de datas e as demais colunas são variáveis exógenas. Entregue apenas valores numéricos dessas previsões no 
formato CSV (com cabeçalho) — nada além disso. Não use qualquer informação que não estivesse disponível 
em {pd.to_datetime("today").strftime("%B %d, %Y")} para formular essas previsões.
"""


# =========== Chamada ao Gemini para gerar previsão via IA
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
modelo_ia = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Upload do arquivo de dados
arquivo = genai.upload_file("dados/pib.csv")

# Lê o CSV retornado pelo Gemini
previsao3 = pd.read_csv(
    filepath_or_buffer=StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
    names=["date", "Valor"],
    skiprows=1,
    index_col="date",
    converters={"date": pd.to_datetime}
).assign(Tipo="IA")


# =========== Salvar previsões
pasta = "previsao"
if not os.path.exists(pasta):
    os.makedirs(pasta)

pd.concat(
    [
        y.rename("Valor").to_frame().assign(Tipo="PIB"),
        previsao1,
        previsao2,
        previsao3
    ]
).to_parquet("previsao/pib.parquet")
