# ==========================================================
# MÓDULO DE PREPARAÇÃO E PREVISÃO DO IPCA
# Objetivo: carregar série mensal, aplicar transformações conforme metadados,
# montar cenários (expectativas, commodities, câmbio, prévia de preços),
# estimar modelos (Ridge e Huber) e gerar previsões também via IA.
# ==========================================================


# =========================
# Bibliotecas (já importadas no escopo principal do projeto)
# - assumimos que ForecasterAutoreg, Ridge, HuberRegressor, PowerTransformer,
#   StringIO, genai, pandas, numpy e os já estão disponíveis.
# =========================
import pandas as pd
import numpy as np
import os, time
from datetime import datetime, timedelta
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge
from sklearn.ensemble import VotingRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PowerTransformer
from io import StringIO
import google.generativeai as genai

# =========================
# Configurações e parâmetros
# =========================
# h: horizonte em meses; inicio_treino: data inicial da amostra; semente: random seed.
h = 12  # horizonte de previsão
inicio_treino = pd.to_datetime("2004-01-01")  # amostra inicial de treinamento
semente = 1984  # semente para reprodução


# =========================
# Função de transformação
# =========================
def transformar(x, tipo):
    """
    Aplica transformação indicada pelo metadado:
    1: nível, 2: primeira diferença, 3: segunda diferença,
    4: log, 5: log + diff, 6: log + diff + diff.
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


# =========================
# Metadados
# =========================
# Lê planilha que indica qual transformação aplicar em cada série
metadados = (
    pd.read_excel(
        io="https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name="Metadados",
        dtype=str,
        index_col="Identificador"
    )
    .filter(["Transformação"])
)


# =========================
# Importa dados mensais já serializados
# =========================
dados_brutos = pd.read_parquet("dados/df_mensal.parquet")


# =========================
# Ajusta frequência e separa Y e X
# =========================
dados_tratados = dados_brutos.asfreq("MS")  # garante frequência mensal regular

# Y = IPCA (variável alvo)
y = dados_tratados.ipca.dropna()

# X = exógenas (todas exceto ipca)
x = dados_tratados.drop(labels="ipca", axis="columns").copy()


# =========================
# Concatena saldo CAGED (antigo + novo) e transforma
# =========================
# Usa combine_first para priorizar antigo quando presente, aplica transformação "5"
x = (
    x
    .assign(saldo_caged = transformar(x.saldo_caged_antigo.combine_first(x.saldo_caged_novo), "5"))
    .drop(labels=["saldo_caged_antigo", "saldo_caged_novo"], axis="columns")
)


# =========================
# Aplica transformações indicadas nos metadados (exceto saldo_caged já tratado)
# =========================
for col in x.drop(labels="saldo_caged", axis="columns").columns.to_list():
    x[col] = transformar(x[col], metadados.loc[col, "Transformação"])


# =========================
# Filtra amostra para início de treino e separa exógenas além de y
# =========================
y = y[y.index >= inicio_treino]
x_alem_de_y = x.query("index >= @y.index.max()")  # exógenas após última observação de y
x = x.query("index >= @inicio_treino and index <= @y.index.max()")  # exógenas no período de treino


# =========================
# Identifica e remove colunas com muitos NAs
# =========================
# Calcula proporção de NA relativa ao nº de observações de y
prop_na = x.isnull().sum() / y.shape[0]

# Remove variáveis com >= 20% de NAs
x = x.drop(labels=prop_na[prop_na >= 0.2].index.to_list(), axis="columns")


# =========================
# Preenche NAs restantes
# =========================
# Estratégia simples: backward fill seguido de forward fill
x = x.bfill().ffill()


# =========================
# Adiciona dummies sazonais (mês) e junta às exógenas
# =========================
dummies_sazonais = (
    pd.get_dummies(y.index.month_name())
    .astype(int)
    .drop(labels="December", axis="columns")  # evita multicolinearidade pela referência
    .set_index(y.index)
)
x = x.join(other=dummies_sazonais, how="outer")


# =========================
# Seleção final de variáveis regressoras (inclui dummies)
# =========================
x_reg = [
    "expec_ipca_top5_curto_prazo",
    "ic_br",
    "cambio_brl_eur",  # Corrigido: era "cambio_bn1_eun"
    "ipc_s"
] + dummies_sazonais.columns.to_list()
# Observação: + 1 lag (definido no ForecasterAutoreg)


# =========================
# Ajusta modelos com amostra completa
# =========================
# Modelo 1: Ridge com 1 lag
modelo1 = ForecasterAutoreg(
    regressor=Ridge(random_state=semente),
    lags=1,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo1.fit(y, x[x_reg])

# Modelo 2: Huber (robusto) com 1 lag
modelo2 = ForecasterAutoreg(
    regressor=HuberRegressor(),
    lags=1,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo2.fit(y, x[x_reg])


# =========================
# Período de previsão fora da amostra (mensal)
# =========================
periodo_previsao = pd.date_range(
    start=modelo1.last_window.index[0] + pd.offsets.MonthBegin(1),
    end=modelo1.last_window.index[0] + pd.offsets.MonthBegin(h),
    freq="MS"
)


# =========================
# Coleta e prepara cenário: expectativas de inflação curto prazo (Focus)
# =========================
dados_focus_exp_ipca = (
    pd.read_csv(
        filepath_or_buffer=f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTop5Mensais?$filter=Indicador%20eq%20'IPCA'%20and%20tipoCalculo%20eq%20'C'%20and%20Data%20ge%20'{periodo_previsao.min().strftime('%Y-%m-%d')}'&$format=text/csv",
        decimal=",",
        converters={
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format="%m/%Y")
        }
    )
)

# Escolhe a data do relatório Focus que tem previsões completas para o horizonte h
data_focus_exp_ipca = (
    dados_focus_exp_ipca
    .query("DataReferencia in @periodo_previsao")
    .Data
    .value_counts()
    .to_frame()
    .reset_index()
    .query("count == @h").query("Data == Data.max()")
    .Data
    .to_list()[0]
)

# Constrói cenário de expectativas (mediana) para o horizonte
dados_cenario_exp_ipca = (
    dados_focus_exp_ipca
    .query("DataReferencia in @periodo_previsao")
    .query("Data == @data_focus_exp_ipca")
    .set_index("DataReferencia")
    .filter(["Mediana"])
    .rename(columns={"Mediana": "expec_ipca_top5_curto_prazo"})
)


# =========================
# Cenário: commodities (ic_br) — mediana por mês projetada para o horizonte
# =========================
dados_cenario_ic_br = (
    x
    .filter(["ic_br"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes=lambda x: x.index.month_name())
    .groupby(["mes"], as_index=False)
    .ic_br
    .median()
    .set_index("mes")
    .join(
        other=(
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes=lambda x: x.data.dt.month_name())
            .drop("data", axis="columns")
            .reset_index()
            .set_index("mes")
        ),
        how="outer"
    )
    .set_index("data")
)


# =========================
# Coleta e constrói cenário do câmbio (cambio_brl_eur)
# =========================
dados_focus_cambio = (
    pd.read_csv(
        filepath_or_buffer=f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTop5Mensais?$filter=Indicador%20eq%20'C%C3%A2mbio'%20and%20tipoCalculo%20eq%20'M'%20and%20Data%20ge%20'{modelo1.last_window.index[0].strftime('%Y-%m-%d')}'&$format=text/csv",
        decimal=",",
        converters={
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format="%m/%Y")
        }
    )
)

# Seleciona a data do relatório Focus apropriada (última com horizonte h+1 quando aplicável)
data_focus_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .Data
    .value_counts()
    .to_frame()
    .reset_index()
    .query("count == @h+1").query("Data == Data.max()")
    .Data
    .to_list()[0]
)

# Constrói cenário de câmbio (mediana) e aplica transformação definida nos metadados
dados_cenario_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .query("Data == @data_focus_cambio")
    .set_index("DataReferencia")
    .filter(["Mediana"])
    .rename(columns={"Mediana": "cambio_brl_eur"})
    .assign(
        cambio_brl_eur=lambda x: transformar(x.cambio_brl_eur, metadados.loc["cambio_brl_eur"].iloc[0])
    )
    .dropna()
)


# =========================
# Cenário: prévia de preços (ipc_s) — mediana por mês projetada
# =========================
dados_cenario_ipc_s = (
    x
    .filter(["ipc_s"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes=lambda x: x.index.month_name())
    .groupby(["mes"], as_index=False)
    .ipc_s
    .median()
    .set_index("mes")
    .join(
        other=(
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes=lambda x: x.data.dt.month_name())
            .drop("data", axis="columns")
            .reset_index()
            .set_index("mes")
        ),
        how="outer"
    )
    .set_index("data")
)


# =========================
# Junta todos os cenários e cria dummies sazonais correspondentes
# =========================
dados_cenarios = (
    dados_cenario_exp_ipca
    .join(
        other=[
            dados_cenario_ic_br,
            dados_cenario_cambio,
            dados_cenario_ipc_s,
            (
                pd.get_dummies(dados_cenario_exp_ipca.index.month_name())
                .astype(int)
                .drop(labels="December", axis="columns")
                .set_index(dados_cenario_exp_ipca.index)
            )
        ],
        how="outer"
    )
)


# =========================
# Produz previsões dos modelos tradicionais
# =========================
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
    .assign(Tipo="Huber")
    .rename(
        columns={
            "pred": "Valor",
            "lower_bound": "Intervalo Inferior",
            "upper_bound": "Intervalo Superior"
        }
    )
)


# =========================
# Gera input CSV para IA (historico ipca + exógenas) e constrói prompt em inglês
# =========================
y.to_frame().join(x[x_reg]).to_csv("dados/ipca.csv")
prompt = f"""
Assume that you are in {pd.to_datetime("today").strftime("%B %d, %Y")}. 
Please give me your best forecast of month-over-month IPCA inflation rate in 
Brazil, published by IBGE, for {periodo_previsao.min().strftime("%B %Y")} to 
{periodo_previsao.max().strftime("%B %Y")}. Use the historical IPCA data from 
the attached CSV file named "ipca.csv", where "ipca" is the target 
column, "data" is the date column and the others are exogenous variables. 
Please give me numeric values for these forecasts, in a CSV like format with 
a header, and nothing more. Do not use any information that was not available 
to you as of {pd.to_datetime("today").strftime("%B %d, %Y")} to formulate these 
forecasts.
"""

# =========================
# Chamada ao Gemini para obter previsão via IA
# =========================
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
modelo_ia = genai.GenerativeModel(model_name="gemini-1.5-pro")
arquivo = genai.upload_file("dados/ipca.csv")

previsao3 = pd.read_csv(
    filepath_or_buffer=StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
    names=["date", "Valor"],
    skiprows=1,
    index_col="date",
    converters={"date": pd.to_datetime}
).assign(Tipo="IA")


# =========================
# Salvar previsões (histórico + modelos + IA) em parquet
# =========================
pasta = "previsao"
if not os.path.exists(pasta):
    os.makedirs(pasta)

pd.concat([
    y.rename("Valor").to_frame().assign(Tipo="IPCA"),
    previsao1,
    previsao2,
    previsao3
]).to_parquet("previsao/ipca.parquet")
