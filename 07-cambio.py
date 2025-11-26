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
h = 12                                          # horizonte de previsão (número de passos à frente)
inicio_treino = pd.to_datetime("2004-01-01")    # data inicial para a amostra de treinamento
seed = 1984                                     # seed para garantir reprodutibilidade


# =========== Função para transformar dados, conforme definido nos metadados
def transformar(x, tipo):
  """
  Aplica transformações padronizadas à série x conforme o código 'tipo' presente em metadados.
  Tipos suportados:
    1: nível (sem transformação)
    2: primeira diferença
    3: segunda diferença
    4: log
    5: log + primeira diferença
    6: log + segunda diferença
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
metadados = (
    pd.read_excel(
        io = "https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name = "Metadados",
        dtype = str,
        index_col = "Identificador"
        )
    .filter(["Transformação"])  # mantemos apenas a coluna com a indicação da transformação
)


# =========== Importa dados online
dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")       # série mensal principal
dados_brutos_t = pd.read_parquet("dados/df_trimestral.parquet")  # séries trimestrais
dados_brutos_a = pd.read_parquet("dados/df_anual.parquet")       # séries anuais


# =========== Converte frequência
dados_tratados = (
    dados_brutos_m
    .asfreq("MS")  # garante frequência mensal com início do mês
    .join(
        other = dados_brutos_a.asfreq("MS").ffill(),  # traz anuais para mensal, preenchendo para frente
        how = "outer"
        )
    .join(
        other = (
            dados_brutos_t
            .filter(["us_gdp", "pib"])
            .dropna()
            # calcula crescimento anual médio móvel de 4 trimestres em %
            .assign(us_gdp = lambda x: ((x.us_gdp.rolling(4).mean() / x.us_gdp.rolling(4).mean().shift(4)) - 1) * 100)
            .asfreq("MS")
            .ffill()
        ),
        how = "outer"
    )
    .rename_axis("data", axis = "index")  # nomeia índice como 'data'
)


# =========== # Separa Y
y = dados_tratados.cambio.dropna()  # variável alvo: câmbio BRL/USD (remove NaNs iniciais)


# =========== Separa X
x = dados_tratados.drop(labels = "cambio", axis = "columns").copy()  # todas as exógenas (cópia para segurança)


# =========== Computa transformações
for col in x.drop(labels = ["saldo_caged_antigo", "saldo_caged_novo"], axis = "columns").columns.to_list():
  # aplica a transformação indicada em metadados para cada coluna (exceto colunas especificadas)
  x[col] = transformar(x[col], metadados.loc[col, "Transformação"])


# =========== Filtra amostra
y = y[y.index >= inicio_treino]  # limita y à amostra a partir de inicio_treino
x_alem_de_y = x.query("index >= @y.index.max()")  # exógenas após a última data observada de y (para horizonte)
x = x.query("index >= @inicio_treino and index <= @y.index.max()")  # exógenas alinhadas com a amostra de y


# =========== Conta por coluna proporção de NAs em relação ao nº de obs. do IPCA
prop_na = x.isnull().sum() / y.shape[0]  # proporção de valores ausentes por coluna (denominador: tamanho de y)


# =========== Remove variáveis que possuem mais de 20% de NAs
x = x.drop(labels = prop_na[prop_na >= 0.2].index.to_list(), axis = "columns")  # descarta colunas com muitos NAs


# =========== Preenche NAs restantes com a vizinhança
x = x.bfill().ffill()  # preenche valores faltantes com método backward e depois forward


# =========== Seleção final de variáveis
x_reg = [
    "selic",
    "expec_cambio",
    "ic_br_agro",
    "cotacao_petroleo_fmi"
    ]
# comentário: estas são as variáveis exógenas finais usadas no modelo (será usado também 1 lag automático)


# =========== Reestima os 2 melhores modelos com amostra completa
modelo1 = ForecasterAutoreg(
    regressor = BayesianRidge(),           # regressor Bayesiano (predição pontual)
    lags = 1,                              # usa 1 lag da variável alvo
    transformer_y = PowerTransformer(),   # transforma y para estabilizar variância
    transformer_exog = PowerTransformer() # transforma exógenas
    )
modelo1.fit(y, x[x_reg])  # encaixa o modelo usando y e as exógenas selecionadas

modelo2 = ForecasterAutoreg(
    regressor = HuberRegressor(),          # regressor robusto (menos sensível a outliers)
    lags = 1,
    transformer_y = PowerTransformer(),
    transformer_exog = PowerTransformer()
    )
modelo2.fit(y, x[x_reg])  # encaixa o segundo modelo


# =========== Período de previsão fora da amostra
periodo_previsao = pd.date_range(
    start = modelo1.last_window.index[0] + pd.offsets.MonthBegin(1),  # mês seguinte à última janela
    end = modelo1.last_window.index[0] + pd.offsets.MonthBegin(h),    # até h meses adiante
    freq = "MS"
    )


# =========== Coleta dados de expectativas da Selic (selic)
dados_focus_selic = (
    pd.read_csv(
        filepath_or_buffer = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTop5Selic?$filter=Data%20ge%20'{modelo1.last_window.index[0].strftime('%Y-%m-%d')}'%20and%20tipoCalculo%20eq%20'C'&$format=text/csv",
        decimal = ",",
        converters = {
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format = "%m/%Y")
            }
        ))
# comentário: traz as expectativas de mercado para a Selic publicadas pelo BCB desde a última data observada


# =========== Constrói cenário para expectativas de juros (selic)
dados_cenario_selic = (
    dados_focus_selic
    .query("Data == Data.max()")  # usa o relatório mais recente disponível da consulta
    .rename(columns = {"mediana": "selic"})  # padroniza nome da coluna para 'selic'
    .head(12)  # pega no máximo 12 observações (horizonte)
    .filter(["selic"])
    .set_index(periodo_previsao)  # indexa pelas datas de previsão
)


# =========== Coleta dados de expectativas do câmbio (expec_cambio)
dados_focus_cambio = (
    pd.read_csv(
        filepath_or_buffer = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?$filter=Indicador%20eq%20'C%C3%A2mbio'%20and%20baseCalculo%20eq%200%20and%20Data%20ge%20'{modelo1.last_window.index[0].strftime('%Y-%m-%d')}'&$format=text/csv",
        decimal = ",",
        converters = {
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format = "%m/%Y")
            }
        ))
# comentário: baixa as expectativas mensais para câmbio reportadas no Focus do BCB


# =========== Data do relatório Focus usada para construir cenário para câmbio
data_focus_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .Data
    .value_counts()
    .to_frame()
    .reset_index()
    .query("count == @h")
    .query("Data == Data.max()")
    .Data
    .to_list()[0]
)
# comentário: seleciona a data do relatório que contém h observações completas para usar como cenário


# =========== Constrói cenário para câmbio (expec_cambio)
dados_cenario_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .query("Data == @data_focus_cambio")
    .sort_values(by = "DataReferencia")
    .set_index("DataReferencia")
    .filter(["Mediana"])
    .rename(columns = {"Mediana": "expec_cambio"})
    .dropna()
)
# comentário: organiza a mediana das expectativas em um índice alinhado ao periodo_previsao


# =========== Constrói cenário para commodities (ic_br_agro)
dados_cenario_ic_br_agro = (
    x
    .filter(["ic_br_agro"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes = lambda x: x.index.month_name())  # extrai nome do mês para sazonalidade
    .groupby(["mes"], as_index = False)
    .ic_br_agro
    .median()  # calcula mediana por mês (perfil sazonal)
    .set_index("mes")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes = lambda x: x.data.dt.month_name())
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("mes")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)
# comentário: gera um cenário mensal para 'ic_br_agro' usando medianas históricas por mês


# =========== Constrói cenário para Petróleo (cotacao_petroleo_fmi)
dados_cenario_cotacao_petroleo_fmi = (
    x
    .filter(["cotacao_petroleo_fmi"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes = lambda x: x.index.month_name())
    .groupby(["mes"], as_index = False)
    .cotacao_petroleo_fmi
    .median()
    .set_index("mes")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes = lambda x: x.data.dt.month_name())
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("mes")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)
# comentário: mesmo procedimento do item anterior, aplicado à cotação do petróleo (FMI)


# =========== Junta cenários e gera dummies sazonais
dados_cenarios = (
    dados_cenario_selic
    .join(
        other = [
            dados_cenario_cambio,
            dados_cenario_ic_br_agro,
            dados_cenario_cotacao_petroleo_fmi
            ],
        how = "outer"
        )
)
# comentário: consolida todos os cenários exógenos em um único DataFrame para previsão


# =========== Produz previsões
previsao1 = (
   modelo1.predict_interval(
      steps = h,
      exog = dados_cenarios,
      n_boot = 5000,
      random_state = seed
      )
    .assign(Tipo = "Bayesian Ridge")  # identifica a origem do modelo
    .rename(
       columns = {
          "pred": "Valor", 
          "lower_bound": "Intervalo Inferior", 
          "upper_bound": "Intervalo Superior"
          }
    )
)

previsao2 = (
   modelo2.predict_interval(
      steps = h,
      exog = dados_cenarios,
      n_boot = 5000,
      random_state = seed
      )
    .assign(Tipo = "Huber")
    .rename(
       columns = {
          "pred": "Valor", 
          "lower_bound": "Intervalo Inferior", 
          "upper_bound": "Intervalo Superior"
          }
    )
)
# comentário: ambos os modelos geram previsão pontual e intervalos de confiança via bootstrap


# salva dados históricos + exógenas em CSV para uso do modelo IA
y.to_frame().join(x[x_reg]).to_csv("dados/cambio.csv")


# =========== Prepara prompt e chama modelo generativo (Gemini) para previsão via IA
prompt = f"""
Considere que a data atual é {pd.to_datetime("today").strftime("%B %d, %Y")}. 
Forneça sua melhor previsão da taxa de câmbio do Brasil, expressa em BRL/USD e divulgada pelo Banco Central do Brasil, para o
período de {periodo_previsao.min().strftime("%B %Y")} a {periodo_previsao.max().strftime("%B %Y")}. 
Utilize os dados históricos presentes no arquivo CSV anexo chamado "cambio.csv", em que a coluna "cambio" é a variável alvo, "data" é 
a coluna de datas e as demais colunas são variáveis exógenas. Entregue apenas valores numéricos dessas previsões no 
formato CSV (com cabeçalho) — nada além disso. Não utilize qualquer informação que não estivesse disponível 
em {pd.to_datetime("today").strftime("%B %d, %Y")} para formular essas previsões.
"""

genai.configure(api_key = os.environ["GEMINI_API_KEY"])  # configura chave via variável de ambiente
modelo_ia = genai.GenerativeModel(model_name = "gemini-1.5-pro")  # instancia o modelo
arquivo = genai.upload_file("dados/cambio.csv")  # faz upload do CSV para o serviço

# chama o modelo generativo e converte a resposta em DataFrame (espera CSV com cabeçalho)
previsao3 = pd.read_csv(
    filepath_or_buffer = StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
    names = ["date", "Valor"],
    skiprows = 1,
    index_col = "date",
    converters = {"date": pd.to_datetime}
    ).assign(Tipo = "IA")
# comentário: previsao3 contém as previsões retornadas pelo modelo de IA, rotuladas como "IA"


# =========== Salvar previsões
pasta = "previsao"
if not os.path.exists(pasta):
  os.makedirs(pasta)  # cria diretório se não existir
  
pd.concat(
    [y.rename("Valor").to_frame().assign(Tipo = "Câmbio"),
    previsao1,
    previsao2,
    previsao3
    ]).to_parquet("previsao/cambio.parquet")  # concatena e salva todas as previsões em parquet
# comentário: saída final com histórico e previsões de todas as fontes, persistida em disco
