# Importa o streamlit para construir a interface web do aplicativo
import streamlit as st
# Importa o pandas para carregar, filtrar e manipular os dados da planilha
import pandas as pd
# Importa o numpy para operações matemáticas e vetoriais simples
import numpy as np
# Importa o altair para criar visualizações e gráficos interativos sofisticados
import altair as alt
# Importa a classe LinearRegression para recriar e aplicar o modelo preditivo
from sklearn.linear_model import LinearRegression

# Configura a página do aplicativo Streamlit para usar o layout expandido (wide)
st.set_page_config(layout="wide")

# Carrega a base de dados em cache para otimizar a velocidade do aplicativo
@st.cache_data
def carregar_dados():
    # Lê o arquivo CSV com os dados consolidados salvos anteriormente
    df = pd.read_csv('DADOS.csv')
    # Retorna o DataFrame carregado para uso no aplicativo
    return df

# Executa a função para carregar a base de dados no aplicativo
dados = carregar_dados()

# Define explicitamente os dicionários de mapeamento conforme configurado anteriormente
map_sazonalidade = {0: 'Inverno', 1: 'Outono', 2: 'Primavera', 3: 'Verão'}
map_clima = {0: 'Chuva forte', 1: 'Chuva leve', 2: 'Neblina', 3: 'Tempo claro'}
map_dia_semana = {0: 'Domingo', 1: 'Quarta-feira', 2: 'Quinta-feira', 3: 'Segunda-feira', 4: 'Sexta-feira', 5: 'Sábado', 6: 'Terça-feira'}

# Inverte os mapeamentos para traduzir o texto selecionado na tela de volta para os códigos numéricos correspondentes
inverso_sazonalidade = {v: k for k, v in map_sazonalidade.items()}
inverso_clima = {v: k for k, v in map_clima.items()}
inverso_dia_semana = {v: k for k, v in map_dia_semana.items()}

# Treina o modelo de regressão dentro do app usando as colunas numéricas adequadas
def treinar_modelo_interno(df):
    # Copia o DataFrame para evitar alterações indesejadas na base original
    df_temp = df.copy()

    # Codifica as variáveis de texto do DataFrame para categorias correspondentes aos códigos numéricos
    for col, mapa in [('sazonalidade', inverso_sazonalidade), ('clima', inverso_clima), ('dia-semana', inverso_dia_semana)]:
        df_temp[col] = df_temp[col].map(mapa)

    # Remove as linhas nulas geradas durante eventuais inconsistências de mapeamento
    df_temp = df_temp.dropna()

    # Define as colunas de recursos para o treinamento excluindo data e alvo
    colunas_excluir = ['dia', 'mês', 'ano', 'alugueis']
    X = df_temp.drop(columns=colunas_excluir)
    # Define a variável alvo para o aprendizado de máquina
    y = df_temp['alugueis']

    # Instancia o algoritmo de regressão linear
    model = LinearRegression()
    # Ajusta o modelo com os dados de entrada
    model.fit(X, y)
    # Retorna o modelo pronto para previsões
    return model

# Armazena o modelo treinado em uma variável global dentro do aplicativo
modelo_treinado = treinar_modelo_interno(dados)

# Adiciona o título principal estilizado no topo da página do aplicativo
st.title("🚴 Previsão de Aluguéis de Bicicletas")

# Divide a tela do aplicativo em duas colunas com larguras iguais para organizar o conteúdo
col1, col2 = st.columns(2)

# Coluna 1: Formulário interativo de entrada de parâmetros
with col1:
    # Cria um cabeçalho para indicar a seção de entrada de dados do simulador
    st.header("⚙️ Parâmetros Simulados")
    # Inicia um formulário Streamlit para agrupar todos os controles interativos do usuário
    with st.form("formulario_parametros"):
        # Cria um seletor para que o usuário informe a sazonalidade atual
        sazonalidade_sel = st.selectbox("Selecione a Estação (Sazonalidade):", list(inverso_sazonalidade.keys()))
        # Cria um seletor para o clima atual
        clima_sel = st.selectbox("Selecione as Condições Climáticas:", list(inverso_clima.keys()))
        # Cria um seletor para o dia da semana atual
        dia_semana_sel = st.selectbox("Selecione o Dia da Semana:", list(inverso_dia_semana.keys()))

        # Cria seletores binários para feriado e dia de trabalho usando caixas de seleção interativas
        feriado_sel = st.checkbox("É Feriado?", value=False)
        dia_trabalho_sel = st.checkbox("É Dia de Trabalho?", value=True)

        # Adiciona sliders interativos para o usuário determinar as condições ambientais exatas
        temperatura_sel = st.slider("Temperatura (°C):", min_value=0.0, max_value=45.0, value=25.0, step=0.5)
        umidade_sel = st.slider("Umidade Relativa do Ar (%):", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        vento_sel = st.slider("Velocidade do Vento (km/h):", min_value=0.0, max_value=50.0, value=15.0, step=0.5)

        # Adiciona o botão de envio do formulário para disparar os cálculos da simulação
        enviar = st.form_submit_button("Calcular Previsão")

# Converte os booleanos selecionados pelo usuário em inteiros (0 ou 1) para o modelo preditivo
feriado_num = 1 if feriado_sel else 0
dia_trabalho_num = 1 if dia_trabalho_sel else 0

# Cria uma linha de dados com as entradas para alimentar o preditor de regressão
dados_entrada = pd.DataFrame([{
    'sazonalidade': inverso_sazonalidade[sazonalidade_sel],
    'feriado': feriado_num,
    'dia-semana': inverso_dia_semana[dia_semana_sel],
    'dia-trabalho': dia_trabalho_num,
    'clima': inverso_clima[clima_sel],
    'temperatura': temperatura_sel,
    'umidade-ar': umidade_sel,
    'velocidade-vento': vento_sel
}])

# Realiza o cálculo da previsão de aluguéis em tempo real e garante que o valor não seja menor que zero
previsao_bruta = modelo_treinado.predict(dados_entrada)[0]
previsao_final = int(max(0, np.round(previsao_bruta)))

# Coluna 2: Cards de Métricas e Comparativos
with col2:
    # Adiciona o título da coluna para as métricas calculadas
    st.header("📊 Resultados da Previsão")

    # Calcula algumas estatísticas básicas da planilha inteira para servir de base comparativa
    media_total = int(np.round(dados['alugueis'].mean()))
    maximo_historico = int(dados['alugueis'].max())

    # Filtra os dados com base na estação selecionada pelo usuário para fazer uma comparação local
    dados_estacao = dados[dados['sazonalidade'] == sazonalidade_sel]
    # Calcula a média de aluguéis especificamente para aquela estação
    media_estacao = int(np.round(dados_estacao['alugueis'].mean())) if len(dados_estacao) > 0 else media_total

    # Cria duas sub-colunas internas para exibir os cards emparelhados de forma harmoniosa
    m_col1, m_col2 = st.columns(2)

    # Exibe a previsão principal estimada pelo modelo no primeiro card da coluna esquerda
    with m_col1:
        st.metric(label="Previsão de Aluguéis", value=f"{previsao_final} bikes", delta=f"{previsao_final - media_total} vs Média Geral")
    # Exibe a média geral registrada na base histórica no segundo card da coluna esquerda
    with m_col2:
        st.metric(label="Média Geral Histórica", value=f"{media_total} bikes")

    # Cria mais duas sub-colunas internas para as métricas adicionais de comparação
    m_col3, m_col4 = st.columns(2)

    # Exibe a média da estação selecionada pelo usuário no primeiro card da segunda linha
    with m_col3:
        st.metric(label=f"Média em {sazonalidade_sel}", value=f"{media_estacao} bikes", delta=f"{previsao_final - media_estacao} vs Previsão")
    # Exibe o recorde absoluto de aluguéis observado na planilha histórica no último card da direita
    with m_col4:
        st.metric(label="Máximo Histórico", value=f"{maximo_historico} bikes")

# Adiciona uma divisória visual antes da seção de gráficos interativos
st.markdown("---")
st.header("📈 Visualizações Gráficas")

# Divide o painel inferior em duas colunas para posicionar os gráficos interativos lado a lado
g_col1, g_col2 = st.columns(2)

# Painel Inferior Esquerdo: Gráfico de Dispersão Temperatura vs Aluguéis com Estrela Amarela
with g_col1:
    st.subheader("Dispersão: Temperatura vs Aluguéis")

    # Cria o gráfico base contendo todos os pontos históricos de temperatura e aluguéis
    grafico_dispersao = alt.Chart(dados).mark_circle(size=60, opacity=0.5, color='teal').encode(
        x=alt.X('temperatura:Q', title='Temperatura (°C)'),
        y=alt.Y('alugueis:Q', title='Quantidade de Aluguéis'),
        tooltip=['temperatura', 'alugueis']
    )

    # Estrutura os dados da simulação atual do usuário para plotagem conjunta
    ponto_simulado = pd.DataFrame([{
        'temperatura': temperatura_sel,
        'alugueis': previsao_final
    }])

    # Cria uma marcação especial em formato de triangulo verde para evidenciar o ponto simulado
    estrela_simulada = alt.Chart(ponto_simulado).mark_point(
        size=300,
        shape='star',
        color='gold',
        filled=True,
        stroke='orange',
        strokeWidth=1.5
    ).encode(
        x='temperatura:Q',
        y='alugueis:Q'
    )

    # Sobrepõe os gráficos combinando os dados gerais com a estrela simulada na mesma tela
    dispersao_final = (grafico_dispersao + estrela_simulada).properties(
        width='container',
        height=350
    )

    # Renderiza o gráfico finalizado de dispersão no painel do Streamlit
    st.altair_chart(dispersao_final, use_container_width=True)

# Painel Inferior Direito: Gráfico de Barras Comparando a Previsão com as Médias Históricas por Estação
with g_col2:
    st.subheader("Comparativo de Aluguéis por Estação")

    # Cria uma tabela resumida contendo as médias históricas agrupadas por estação
    medias_por_estacao = dados.groupby('sazonalidade')['alugueis'].mean().reset_index()
    # Altera os nomes de colunas no resumo para facilitar a identificação
    medias_por_estacao.columns = ['Estação', 'Aluguéis']

    # Constrói uma nova linha contendo os dados previstos no cenário atual para inclusão no gráfico
    linha_previsao = pd.DataFrame([{
        'Estação': 'Previsão Simulada',
        'Aluguéis': previsao_final
    }])

    # Junta os dados históricos calculados com a linha do ponto previsto atual
    dados_barras_completos = pd.concat([medias_por_estacao, linha_previsao], ignore_index=True)

    # Define as cores do gráfico destacando a previsão com cor diferente para fins de contraste
    dados_barras_completos['Tipo'] = dados_barras_completos['Estação'].apply(
        lambda x: 'Simulado' if x == 'Previsão Simulada' else 'Histórico'
    )

    # Constrói o gráfico de barras comparando as médias e a previsão simulada lado a lado
    grafico_barras = alt.Chart(dados_barras_completos).mark_bar().encode(
        x=alt.X('Estação:N', sort=None, title='Estação / Cenário'),
        y=alt.Y('Aluguéis:Q', title='Aluguéis de Bicicletas'),
        color=alt.Color('Tipo:N', scale=alt.Scale(
            domain=['Histórico', 'Simulado'],
            range=['#1f77b4', '#ff7f0e']
        ), legend=None),
        tooltip=['Estação', 'Aluguéis']
    ).properties(
        width='container',
        height=350
    )

    # Renderiza o gráfico de barras comparativo na interface web
    st.altair_chart(grafico_barras, use_container_width=True)
