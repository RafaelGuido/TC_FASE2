import streamlit as st
import cv2
import numpy as np
import random
import copy
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configuração da página Streamlit
st.set_page_config(page_title="Visualização do Algoritmo Genético", layout="wide")

# Título da aplicação
st.title(
    "Visualização da Evolução do Algoritmo Genético para Processamento de Captchas"
)


# Função para calcular a similaridade entre duas imagens
def calcular_similaridade(img1, img2):
    # Garantir que as imagens têm o mesmo tamanho
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calcular a similaridade usando o método de correlação normalizada
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]


# Função para processar a imagem com os parâmetros fornecidos
def processar_imagem(params, imagem_path):
    # Extrair parâmetros
    threshold_value = params["threshold"]
    blur_size = params["blur"]
    dilate_kernel_size = params["dilate_size"]
    dilate_kernel_shape = params["dilate_shape"]
    erode_kernel_size = params["erode_size"]
    erode_kernel_shape = params["erode_shape"]

    # Carregar a imagem
    image = cv2.imread(imagem_path)

    # Verificar se a imagem foi carregada corretamente
    if image is None:
        st.error(f"Erro ao carregar a imagem: {imagem_path}")
        return None

    # Aplicar blur
    image = cv2.blur(image, (blur_size, blur_size))

    # Aplicar threshold
    ret, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Aplicar dilate
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_shape), np.uint8)
    image = cv2.dilate(image, dilate_kernel)

    # Aplicar erode
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_shape), np.uint8)
    image = cv2.erode(image, erode_kernel)

    return image


# Função para avaliar a aptidão de um indivíduo
def avaliar_aptidao(individuo, imagem_path, imagem_alvo_path):
    # Processar a imagem com os parâmetros do indivíduo
    imagem_processada = processar_imagem(individuo, imagem_path)

    # Verificar se a imagem foi processada corretamente
    if imagem_processada is None:
        return -1  # Retornar um valor baixo para indicar falha

    # Carregar a imagem alvo
    imagem_alvo = cv2.imread(imagem_alvo_path)

    # Verificar se a imagem alvo foi carregada corretamente
    if imagem_alvo is None:
        st.error(f"Erro ao carregar a imagem alvo: {imagem_alvo_path}")
        return -1  # Retornar um valor baixo para indicar falha

    # Calcular a similaridade
    return calcular_similaridade(imagem_processada, imagem_alvo)


# Função para criar um indivíduo aleatório
def criar_individuo():
    return {
        "threshold": random.randint(50, 150),
        "blur": random.randint(1, 5),
        "dilate_size": random.randint(1, 5),
        "dilate_shape": random.randint(1, 5),
        "erode_size": random.randint(1, 5),
        "erode_shape": random.randint(1, 5),
    }


# Função para criar a população inicial
def criar_populacao(tamanho):
    return [criar_individuo() for _ in range(tamanho)]


# Função para selecionar indivíduos para reprodução (torneio)
def selecionar(populacao, aptidoes, k=3):
    # Selecionar k indivíduos aleatórios e escolher o melhor
    indices_selecionados = random.sample(range(len(populacao)), k)
    indice_melhor = max(indices_selecionados, key=lambda i: aptidoes[i])
    return populacao[indice_melhor]


# Função para cruzar dois indivíduos
def cruzar(pai1, pai2):
    filho = {}
    for param in pai1.keys():
        # 50% de chance de herdar de cada pai
        if random.random() < 0.5:
            filho[param] = pai1[param]
        else:
            filho[param] = pai2[param]
    return filho


# Função para aplicar mutação em um indivíduo
def mutar(individuo, taxa_mutacao=0.2):
    novo_individuo = copy.deepcopy(individuo)
    for param in novo_individuo.keys():
        # Aplicar mutação com probabilidade taxa_mutacao
        if random.random() < taxa_mutacao:
            if param == "threshold":
                novo_individuo[param] = random.randint(50, 150)
            elif param == "blur":
                novo_individuo[param] = random.randint(1, 5)
            elif param in ["dilate_size", "dilate_shape", "erode_size", "erode_shape"]:
                novo_individuo[param] = random.randint(1, 5)
    return novo_individuo


# Função para garantir que a pasta de resultados existe
def garantir_pasta_resultados(pasta="resultados"):
    # Obter caminho absoluto da pasta de resultados
    pasta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pasta)

    # Criar a pasta se não existir
    if not os.path.exists(pasta_path):
        os.makedirs(pasta_path)
        st.success(f"Pasta '{pasta}' criada com sucesso.")

    return pasta_path


# Função para salvar a imagem com os parâmetros e informações
def salvar_imagem(imagem, params, aptidao, captcha_nome, nome_base="resultado"):
    # Garantir que a pasta de resultados existe
    pasta_resultados = garantir_pasta_resultados()

    # Criar nome de arquivo com timestamp e nome do captcha
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{nome_base}_{captcha_nome}_{timestamp}.png"
    caminho_completo = os.path.join(pasta_resultados, nome_arquivo)

    # Salvar a imagem
    cv2.imwrite(caminho_completo, imagem)

    # Salvar informações em um arquivo de texto
    info_arquivo = f"{nome_base}_{captcha_nome}_{timestamp}_info.txt"
    caminho_info = os.path.join(pasta_resultados, info_arquivo)

    with open(caminho_info, "w") as f:
        f.write(f"Captcha: {captcha_nome}\n")
        f.write(f"Aptidão: {aptidao}\n")
        f.write(f"Parâmetros:\n")
        for param, valor in params.items():
            f.write(f"  {param}: {valor}\n")

    st.success(f"Imagem salva como: {caminho_completo}")
    st.success(f"Informações salvas como: {caminho_info}")

    return nome_arquivo


# Algoritmo genético principal com visualização Streamlit
def algoritmo_genetico_streamlit(
    imagem_path,
    imagem_alvo_path,
    captcha_nome,
    tamanho_populacao=20,
    geracoes=50,
    taxa_mutacao=0.2,
):
    # Criar containers para visualização
    header = st.container()
    with header:
        st.subheader(f"Processando captcha: {captcha_nome}")
        col1, col2, col3 = st.columns(3)

        # Carregar e exibir imagens originais
        imagem_original = cv2.imread(imagem_path)
        imagem_alvo = cv2.imread(imagem_alvo_path)

        # Converter de BGR para RGB para exibição no Streamlit
        imagem_original_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        imagem_alvo_rgb = cv2.cvtColor(imagem_alvo, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(
                imagem_original_rgb, caption="Imagem Original", use_container_width=True
            )
        with col3:
            st.image(imagem_alvo_rgb, caption="Imagem Alvo", use_container_width=True)

    # Container para métricas
    metrics = st.container()
    with metrics:
        st.subheader("Métricas de Evolução")
        col1, col2 = st.columns(2)
        with col1:
            geracao_text = st.empty()
            melhor_aptidao_text = st.empty()
            aptidao_media_text = st.empty()
        with col2:
            melhor_params_text = st.empty()

    # Container para gráficos
    charts = st.container()
    with charts:
        st.subheader("Gráficos de Evolução")
        col1, col2 = st.columns(2)
        with col1:
            aptidao_chart = st.empty()
        with col2:
            melhor_imagem_display = st.empty()

    # Criar população inicial
    populacao = criar_populacao(tamanho_populacao)

    # Melhor indivíduo global
    melhor_individuo = None
    melhor_aptidao = -float("inf")
    melhor_imagem = None

    # Dados para gráficos
    dados_evolucao = {
        "geracao": [],
        "melhor_aptidao": [],
        "aptidao_media": [],
    }

    # Evolução por gerações
    for geracao in range(geracoes):
        # Atualizar texto da geração atual
        geracao_text.text(f"Geração: {geracao+1}/{geracoes}")

        # Avaliar aptidão de cada indivíduo
        aptidoes = [
            avaliar_aptidao(ind, imagem_path, imagem_alvo_path) for ind in populacao
        ]

        # Encontrar o melhor indivíduo da geração atual
        indice_melhor = aptidoes.index(max(aptidoes))
        melhor_atual = populacao[indice_melhor]
        aptidao_melhor_atual = aptidoes[indice_melhor]
        aptidao_media = sum(aptidoes) / len(aptidoes)

        # Atualizar métricas
        melhor_aptidao_text.text(
            f"Melhor aptidão da geração: {aptidao_melhor_atual:.4f}"
        )
        aptidao_media_text.text(f"Aptidão média da geração: {aptidao_media:.4f}")
        melhor_params_text.text(f"Melhores parâmetros:\n{melhor_atual}")

        # Atualizar dados para gráficos
        dados_evolucao["geracao"].append(geracao + 1)
        dados_evolucao["melhor_aptidao"].append(aptidao_melhor_atual)
        dados_evolucao["aptidao_media"].append(aptidao_media)

        # Plotar gráfico de evolução
        df = pd.DataFrame(dados_evolucao)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["geracao"], df["melhor_aptidao"], label="Melhor Aptidão", marker="o")
        ax.plot(df["geracao"], df["aptidao_media"], label="Aptidão Média", marker="x")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Aptidão")
        ax.set_title("Evolução da Aptidão ao Longo das Gerações")
        ax.legend()
        ax.grid(True)
        aptidao_chart.pyplot(fig)
        plt.close(fig)

        # Atualizar o melhor global se necessário
        if aptidao_melhor_atual > melhor_aptidao:
            melhor_individuo = copy.deepcopy(melhor_atual)
            melhor_aptidao = aptidao_melhor_atual
            melhor_imagem = processar_imagem(melhor_individuo, imagem_path)

            # Exibir a melhor imagem encontrada até agora
            melhor_imagem_rgb = cv2.cvtColor(melhor_imagem, cv2.COLOR_BGR2RGB)
            melhor_imagem_display.image(
                melhor_imagem_rgb,
                caption=f"Melhor Resultado (Aptidão: {melhor_aptidao:.4f})",
                use_container_width=True,
            )

        # Criar nova população
        nova_populacao = []

        # Elitismo: manter o melhor indivíduo
        nova_populacao.append(melhor_atual)

        # Gerar o resto da população
        while len(nova_populacao) < tamanho_populacao:
            # Seleção
            pai1 = selecionar(populacao, aptidoes)
            pai2 = selecionar(populacao, aptidoes)

            # Cruzamento
            filho = cruzar(pai1, pai2)

            # Mutação
            filho = mutar(filho, taxa_mutacao)

            nova_populacao.append(filho)

        # Substituir a população antiga pela nova
        populacao = nova_populacao

        # Pequena pausa para visualização
        time.sleep(0.1)

    st.success("\nProcessamento concluído!")
    st.success(f"Melhor aptidão encontrada: {melhor_aptidao:.4f}")
    st.json(melhor_individuo)

    # Processar a melhor imagem encontrada (caso ainda não tenha sido processada)
    if melhor_imagem is None:
        melhor_imagem = processar_imagem(melhor_individuo, imagem_path)

    # Salvar a melhor imagem encontrada
    nome_arquivo = salvar_imagem(
        melhor_imagem, melhor_individuo, melhor_aptidao, captcha_nome
    )

    # Exibir as imagens finais lado a lado
    final_col1, final_col2, final_col3 = st.columns(3)
    with final_col1:
        st.image(
            imagem_original_rgb, caption="Imagem Original", use_container_width=True
        )
    with final_col2:
        melhor_imagem_rgb = cv2.cvtColor(melhor_imagem, cv2.COLOR_BGR2RGB)
        st.image(
            melhor_imagem_rgb, caption="Melhor Resultado", use_container_width=True
        )
    with final_col3:
        st.image(imagem_alvo_rgb, caption="Imagem Alvo", use_container_width=True)

    return melhor_individuo, melhor_aptidao, nome_arquivo


# Função para processar captchas com interface Streamlit
def processar_captchas_streamlit(
    pasta_imgs="imgs", tamanho_populacao=20, geracoes=50, taxa_mutacao=0.2
):
    # Obter caminho absoluto da pasta imgs
    pasta_imgs_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), pasta_imgs
    )

    # Verificar se a pasta existe
    if not os.path.exists(pasta_imgs_path):
        st.error(f"Erro: A pasta {pasta_imgs_path} não existe!")
        return

    # Encontrar todos os arquivos captcha*.png que não contêm 'target' no nome
    captchas = []
    for arquivo in os.listdir(pasta_imgs_path):
        if (
            arquivo.startswith("captcha")
            and arquivo.endswith(".png")
            and "target" not in arquivo
        ):
            base_name = arquivo.split(".")[0]  # Remove a extensão
            target_file = f"{base_name}_target.png"

            # Verificar se o arquivo target correspondente existe
            if os.path.exists(os.path.join(pasta_imgs_path, target_file)):
                captchas.append((base_name, arquivo, target_file))

    # Verificar se encontrou captchas
    if not captchas:
        st.error("Nenhum par de captcha/target encontrado na pasta!")
        return

    # Exibir seletor de captcha
    opcoes_captcha = [base_name for base_name, _, _ in captchas]
    captcha_selecionado = st.selectbox(
        "Selecione o captcha para processar:", opcoes_captcha, index=0
    )

    # Configurações do algoritmo genético
    st.subheader("Configurações do Algoritmo Genético")
    col1, col2, col3 = st.columns(3)
    with col1:
        tamanho_pop = st.slider("Tamanho da População", 10, 100, tamanho_populacao)
    with col2:
        num_geracoes = st.slider("Número de Gerações", 10, 200, geracoes)
    with col3:
        taxa_mut = st.slider("Taxa de Mutação", 0.0, 1.0, taxa_mutacao)

    # Botão para iniciar o processamento
    if st.button("Iniciar Processamento"):
        # Encontrar o captcha selecionado
        for base_name, captcha_file, target_file in captchas:
            if base_name == captcha_selecionado:
                # Caminhos completos para os arquivos
                captcha_path = os.path.join(pasta_imgs_path, captcha_file)
                target_path = os.path.join(pasta_imgs_path, target_file)

                # Executar o algoritmo genético para este par
                melhores_params, melhor_similaridade, arquivo_salvo = (
                    algoritmo_genetico_streamlit(
                        captcha_path,
                        target_path,
                        base_name,
                        tamanho_populacao=tamanho_pop,
                        geracoes=num_geracoes,
                        taxa_mutacao=taxa_mut,
                    )
                )

                # Exibir resultados finais
                st.subheader("Resultados Finais")
                st.write(f"Captcha: {base_name}")
                st.write(f"Similaridade: {melhor_similaridade:.4f}")
                st.write(f"Arquivo salvo: {arquivo_salvo}")
                st.json(melhores_params)
                break


# Função para calcular a média de parâmetros de múltiplos resultados
def calcular_media_parametros(resultados):
    if not resultados or len(resultados) == 0:
        return None

    # Inicializar dicionário para somar os valores
    soma_params = {}
    for param in resultados[0]["params"].keys():
        soma_params[param] = 0

    # Somar todos os valores
    for resultado in resultados:
        for param, valor in resultado["params"].items():
            soma_params[param] += valor

    # Calcular a média
    media_params = {}
    for param, soma in soma_params.items():
        # Arredondar para o inteiro mais próximo, já que os parâmetros são inteiros
        media_params[param] = round(soma / len(resultados))

    return media_params


# Interface principal do Streamlit
def main():
    st.sidebar.title("Opções")
    opcao = st.sidebar.radio(
        "Selecione uma opção:", ["Processar Captchas", "Sobre o Algoritmo"]
    )

    if opcao == "Processar Captchas":
        processar_captchas_streamlit()
    elif opcao == "Sobre o Algoritmo":
        st.header("Sobre o Algoritmo Genético")
        st.write(
            """
        Este aplicativo utiliza um algoritmo genético para otimizar o processamento de imagens de captcha.
        
        O algoritmo funciona da seguinte forma:
        1. **Inicialização**: Uma população inicial de indivíduos (conjuntos de parâmetros) é criada aleatoriamente.
        2. **Avaliação**: Cada indivíduo é avaliado calculando a similaridade entre a imagem processada e a imagem alvo.
        3. **Seleção**: Os melhores indivíduos são selecionados para reprodução.
        4. **Cruzamento**: Novos indivíduos são criados combinando os parâmetros dos pais.
        5. **Mutação**: Pequenas alterações aleatórias são aplicadas aos novos indivíduos.
        6. **Substituição**: A nova geração substitui a antiga, e o processo se repete.
        
        Os parâmetros otimizados incluem:
        - **Threshold**: Valor de limiarização para binarização da imagem.
        - **Blur**: Tamanho do kernel para desfoque.
        - **Dilate Size/Shape**: Tamanho e forma do kernel para dilatação.
        - **Erode Size/Shape**: Tamanho e forma do kernel para erosão.
        """
        )


# Executar a aplicação quando o script é executado diretamente
if __name__ == "__main__":
    main()
