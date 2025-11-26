# ==========================================================
# MÓDULO DE COLETA ROBUSTA DE SÉRIES TEMPORAIS
# Objetivo: funções utilitárias para baixar séries de múltiplas APIs
# (BCB SGS, ODATA, IPEADATA, SIDRA, FRED, IFI) com tentativas
# automáticas e fragmentação de intervalos longos.
# ==========================================================

# Retenta ler um CSV se falhar download
def ler_csv(*args, **kwargs):
  """
  Tenta ler um CSV via pd.read_csv até 'max_tentativas' vezes.
  Em caso de falha espera 'intervalo' segundos entre tentativas.
  Retorna DataFrame lido ou None se todas as tentativas falharem.
  """
  max_tentativas = 5
  intervalo = 2
  tentativas = 0
  while tentativas < max_tentativas:
      try:
          df = pd.read_csv(*args, **kwargs)   # tenta fazer o download/leitura
          return df                          # retorna assim que obtém sucesso
      except Exception as e:
          tentativas += 1
          print(f"Tentativa {tentativas} falhou: {e}")  # log simples da falha
          time.sleep(intervalo)                         # espera antes de nova tentativa
  print(f"Falha após {max_tentativas} tentativas.")
  return None


# Coleta dados da API do Banco Central (SGS)
def coleta_bcb_sgs(codigo, nome, freq, data_inicio = "01/01/2000", data_fim = (pd.to_datetime("today") + pd.offsets.DateOffset(months = 36)).strftime("%d/%m/%Y")):
  """
  Coleta uma série do SGS do BCB. Se a frequência for 'Diária' divide o
  intervalo em janelas (para evitar limites de download) e concatena os blocos.
  Retorna DataFrame com índice datetime e coluna renomeada para 'nome'.
  """
  if freq == "Diária":
    datas_inicio = split_date_range(data_inicio, data_fim)  # divide intervalo em janelas
  else:
    datas_inicio = [(data_inicio, data_fim)]

  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = []
    for d in datas_inicio:
      url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv&dataInicial={d[0]}&dataFinal={d[1]}"
      resposta.append(ler_csv(filepath_or_buffer = url, sep = ";", decimal = ","))  # usa a função resiliente
    resposta = pd.concat(resposta)  # concatena os blocos coletados
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    return (
        resposta
        .rename(columns = {"valor": nome})                             # padroniza nome da coluna
        .assign(data = lambda x: pd.to_datetime(x.data, format = "%d/%m/%Y"))  # converte datas
        .set_index("data")                                             # seta índice temporal
    )


# Coleta dados da API do Banco Central (ODATA)
def coleta_bcb_odata(codigo, nome):
  """
  Faz leitura via ODATA (URL completa passada em 'codigo').
  Espera coluna 'Mediana' no retorno e renomeia para 'nome'.
  """
  url = codigo

  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = ler_csv(
        filepath_or_buffer = url,
        sep = ",", decimal = ",",
        converters = {"Data": lambda x: pd.to_datetime(x)}
        )
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    return resposta.rename(columns = {"Mediana": nome})


# Coleta dados da API do IPEA (IPEADATA)
def coleta_ipeadata(codigo, nome):
  """
  Consulta a API OData do IPEA e transforma o JSON retornado em DataFrame
  com colunas 'data' e o nome da série.
  """
  url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo}')"
  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = pd.read_json(url)
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    return (
        pd.DataFrame.from_records(resposta["value"])
        .rename(columns = {"VALVALOR": nome, "VALDATA": "data"})
        .filter(["data", nome])
      )


# Coleta dados da API do IBGE (SIDRA)
def coleta_ibge_sidra(codigo, nome):
  """
  Consulta endpoints do SIDRA que devolvem JSON. Ajusta nomes das colunas,
  filtra valores não numéricos e converte para numérico.
  """
  url = f"{codigo}?formato=json"
  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = pd.read_json(url)
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    df = (
        resposta
        .rename(columns = {"D3C": "data", "V": nome})
        .filter(["data", nome])
      )
    df = df[-df[nome].isin(["Valor", "...", "-"])]  # remove rótulos/valores inválidos
    df[nome] = pd.to_numeric(df[nome])              # força tipo numérico
    return df


# Coleta dados da API do FRED
def coleta_fred(codigo, nome):
  """
  Baixa CSV público do FRED para a série 'codigo' e renomeia colunas
  para 'data' e o nome da série.
  """
  url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={codigo}"

  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = ler_csv(
        filepath_or_buffer = url,
        converters = {"DATE": lambda x: pd.to_datetime(x)}
        )
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    return resposta.rename(columns = {"DATE": "data", codigo: nome})


# Coleta dados via link da IFI
def coleta_ifi(codigo, nome):
  """
  Lê planilha Excel (link/arquivo em 'codigo') na aba 'Hiato do Produto'
  e retorna colunas de interesse com possível estrutura de confiança.
  """
  try:
    print(f"Coletando a série {codigo} ({nome})")
    resposta = pd.read_excel(
        io = codigo,
        sheet_name = "Hiato do Produto",
        names = ["data", "lim_inf", nome, "lim_sup"],
        skiprows = 2
        )
  except:
    raise Exception(f"Falha na coleta da série {codigo} ({nome})")
  else:
    return resposta


# Separa intervalo de datas em janelas de 10 anos para coleta de dados em blocos
# na API do BCB/SGS
def split_date_range(start_date_str, end_date_str, interval_years=5):
  """
  Recebe datas no formato 'dd/mm/YYYY' e retorna lista de tuplas
  (data_inicial, data_final) segmentando o período em blocos de X anos.
  Uso: evitar timeouts/limites ao baixar séries diárias muito longas.
  """
  start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
  end_date = datetime.strptime(end_date_str, "%d/%m/%Y")

  result = []
  current_start = start_date

  while current_start < end_date:
    try:
      current_end = current_start.replace(year=current_start.year + interval_years)
    except ValueError:
      current_end = current_start + timedelta(days=365 * interval_years)

    if current_end > end_date:
      current_end = end_date

    result.append((
      current_start.strftime("%d/%m/%Y"),
      current_end.strftime("%d/%m/%Y")
    ))
    current_start = current_end

  return result
