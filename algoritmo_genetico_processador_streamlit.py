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
import glob
import shutil

# Configuração da página Streamlit
st.set_page_config(page_title="Processador de Captchas com Algoritmo Genético", layout="wide")

# Título da aplicação
st.title("Processador de Captchas com Algoritmo Genético")


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


# Algoritmo genético com interface Streamlit
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

        # Placeholder para o gráfico de aptidão
        with col1:
            aptidao_chart = st.empty()

        # Placeholder para os parâmetros atuais
        with col2:
            params_chart = st.empty()

    # Container para a imagem processada
    processed = st.container()
    with processed:
        st.subheader("Imagem Processada (Melhor Atual)")
        processed_img = st.empty()

    # Criar população inicial
    populacao = criar_populacao(tamanho_populacao)

    # Melhor indivíduo global
    melhor_individuo = None
    melhor_aptidao = -float("inf")
    melhor_imagem = None

    # Dados para o gráfico
    dados_grafico = {
        "geracao": [],
        "aptidao_media": [],
        "aptidao_melhor": [],
    }

    # Evolução por gerações
    progress_bar = st.progress(0)
    status_text = st.empty()

    for geracao in range(geracoes):
        status_text.text(f"Processando geração {geracao+1}/{geracoes}")

        # Avaliar aptidão de cada indivíduo
        aptidoes = [
            avaliar_aptidao(ind, imagem_path, imagem_alvo_path) for ind in populacao
        ]

        # Encontrar o melhor indivíduo da geração atual
        indice_melhor = aptidoes.index(max(aptidoes))
        melhor_atual = populacao[indice_melhor]
        aptidao_melhor_atual = aptidoes[indice_melhor]

        # Calcular aptidão média
        aptidao_media = sum(aptidoes) / len(aptidoes)

        # Atualizar dados do gráfico
        dados_grafico["geracao"].append(geracao + 1)
        dados_grafico["aptidao_media"].append(aptidao_media)
        dados_grafico["aptidao_melhor"].append(aptidao_melhor_atual)

        # Atualizar o gráfico de aptidão
        df = pd.DataFrame(dados_grafico)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["geracao"], df["aptidao_melhor"], "b-", label="Melhor Aptidão")
        ax.plot(df["geracao"], df["aptidao_media"], "r-", label="Aptidão Média")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Aptidão")
        ax.legend()
        ax.grid(True)
        aptidao_chart.pyplot(fig)
        plt.close(fig)

        # Atualizar o gráfico de parâmetros
        params_data = pd.DataFrame([melhor_atual])
        params_chart.write("Melhores parâmetros atuais:")
        params_chart.json(melhor_atual)
        params_chart.write(f"Aptidão: {aptidao_melhor_atual:.4f}")

        # Atualizar o melhor global se necessário
        if aptidao_melhor_atual > melhor_aptidao:
            melhor_individuo = copy.deepcopy(melhor_atual)
            melhor_aptidao = aptidao_melhor_atual
            melhor_imagem = processar_imagem(melhor_individuo, imagem_path)

            # Exibir a melhor imagem atual
            if melhor_imagem is not None:
                melhor_imagem_rgb = cv2.cvtColor(melhor_imagem, cv2.COLOR_BGR2RGB)
                processed_img.image(
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

        # Atualizar a barra de progresso
        progress_bar.progress((geracao + 1) / geracoes)

    # Limpar a barra de progresso e o texto de status
    progress_bar.empty()
    status_text.empty()

    st.success("Processamento concluído!")

    # Exibir resultados finais
    st.subheader("Resultado Final")
    st.write(f"Melhor aptidão encontrada: {melhor_aptidao:.4f}")
    st.write("Melhores parâmetros:")
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
        st.image(imagem_original_rgb, caption="Imagem Original", use_container_width=True)
    with final_col2:
        melhor_imagem_rgb = cv2.cvtColor(melhor_imagem, cv2.COLOR_BGR2RGB)
        st.image(
            melhor_imagem_rgb, caption="Melhor Resultado", use_container_width=True
        )
    with final_col3:
        st.image(imagem_alvo_rgb, caption="Imagem Alvo", use_container_width=True)

    return melhor_individuo, melhor_aptidao, nome_arquivo


# Função para processar todos os captchas com interface Streamlit
def processar_captchas_streamlit(
    pasta_imgs="imgs", tamanho_populacao=20, geracoes=50, taxa_mutacao=0.2, mostrar_config=True
):
    # Obter caminho absoluto da pasta imgs
    pasta_imgs_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), pasta_imgs
    )

    # Verificar se a pasta existe
    if not os.path.exists(pasta_imgs_path):
        st.error(f"Erro: A pasta {pasta_imgs_path} não existe!")
        return None, None

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
        return None, None

    # Configurações do algoritmo genético (apenas se mostrar_config for True)
    if mostrar_config:
        st.subheader("Configurações do Algoritmo Genético")
        col1, col2, col3 = st.columns(3)
        with col1:
            tamanho_pop = st.slider("Tamanho da População", 10, 100, tamanho_populacao, key="captcha_pop")
        with col2:
            num_geracoes = st.slider("Número de Gerações", 10, 200, geracoes, key="captcha_gen")
        with col3:
            taxa_mut = st.slider("Taxa de Mutação", 0.0, 1.0, taxa_mutacao, key="captcha_mut")

        # Opção para processar todos os captchas ou apenas um
        processar_todos = st.checkbox("Processar todos os captchas", value=True, key="captcha_todos")

        if not processar_todos:
            # Exibir seletor de captcha
            opcoes_captcha = [base_name for base_name, _, _ in captchas]
            captcha_selecionado = st.selectbox(
                "Selecione o captcha para processar:", opcoes_captcha, index=0, key="captcha_select"
            )
    else:
        # Usar os parâmetros passados diretamente
        tamanho_pop = tamanho_populacao
        num_geracoes = geracoes
        taxa_mut = taxa_mutacao
        processar_todos = True  # No fluxo completo, sempre processar todos

    # Botão para iniciar o processamento (apenas se mostrar_config for True)
    if mostrar_config:
        processar_agora = st.button("Iniciar Processamento", key="captcha_process_btn")
    else:
        processar_agora = True  # No fluxo completo, processar automaticamente
    
    if processar_agora:
        resultados = []

        if processar_todos:
            # Processar todos os captchas
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (base_name, captcha_file, target_file) in enumerate(captchas):
                status_text.text(f"Processando captcha {i+1}/{len(captchas)}: {base_name}")

                # Caminhos completos para os arquivos
                captcha_path = os.path.join(pasta_imgs_path, captcha_file)
                target_path = os.path.join(pasta_imgs_path, target_file)

                # Executar o algoritmo genético para este par
                melhores_params, melhor_similaridade, arquivo_salvo = algoritmo_genetico_streamlit(
                    captcha_path,
                    target_path,
                    base_name,
                    tamanho_populacao=tamanho_pop,
                    geracoes=num_geracoes,
                    taxa_mutacao=taxa_mut,
                )

                # Armazenar resultados
                resultados.append(
                    {
                        "captcha": base_name,
                        "params": melhores_params,
                        "similaridade": melhor_similaridade,
                        "arquivo": arquivo_salvo,
                    }
                )

                # Atualizar a barra de progresso
                progress_bar.progress((i + 1) / len(captchas))

            # Limpar a barra de progresso e o texto de status
            progress_bar.empty()
            status_text.empty()

            # Calcular a média dos parâmetros
            params_media = calcular_media_parametros(resultados)

            # Exibir resumo dos resultados
            st.subheader("Resumo dos Resultados")
            for resultado in resultados:
                st.write(f"Captcha: {resultado['captcha']}")
                st.write(f"Similaridade: {resultado['similaridade']:.4f}")
                st.write(f"Arquivo salvo: {resultado['arquivo']}")
                st.write("---")

            # Exibir parâmetros médios
            st.subheader("Parâmetros Médios")
            st.json(params_media)

            # Retornar os resultados e os parâmetros médios
            return resultados, params_media

        else:
            # Processar apenas o captcha selecionado
            for base_name, captcha_file, target_file in captchas:
                if base_name == captcha_selecionado:
                    # Caminhos completos para os arquivos
                    captcha_path = os.path.join(pasta_imgs_path, captcha_file)
                    target_path = os.path.join(pasta_imgs_path, target_file)

                    # Executar o algoritmo genético para este par
                    melhores_params, melhor_similaridade, arquivo_salvo = algoritmo_genetico_streamlit(
                        captcha_path,
                        target_path,
                        base_name,
                        tamanho_populacao=tamanho_pop,
                        geracoes=num_geracoes,
                        taxa_mutacao=taxa_mut,
                    )

                    # Armazenar resultados
                    resultados.append(
                        {
                            "captcha": base_name,
                            "params": melhores_params,
                            "similaridade": melhor_similaridade,
                            "arquivo": arquivo_salvo,
                        }
                    )

                    # Exibir resultados finais
                    st.subheader("Resultados Finais")
                    st.write(f"Captcha: {base_name}")
                    st.write(f"Similaridade: {melhor_similaridade:.4f}")
                    st.write(f"Arquivo salvo: {arquivo_salvo}")
                    st.json(melhores_params)
                    break

            # Retornar os resultados
            return resultados, None
    
    # Se o botão não foi clicado, retornar valores padrão
    return None, None


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


# Função para processar imagens da pasta samples usando os melhores parâmetros encontrados
def processar_samples_streamlit(params, pasta_samples="samples", limite_arquivos=None):
    st.subheader(f"Processando imagens da pasta {pasta_samples}")

    # Garantir que a pasta de resultados existe
    pasta_resultados = garantir_pasta_resultados()

    # Obter caminho absoluto da pasta samples
    pasta_samples_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), pasta_samples
    )

    # Verificar se a pasta existe
    if not os.path.exists(pasta_samples_path):
        st.error(f"Erro: A pasta {pasta_samples_path} não existe!")
        return

    # Listar todos os arquivos .png e .jpg na pasta samples
    arquivos_imagem = []
    for arquivo in os.listdir(pasta_samples_path):
        if arquivo.lower().endswith((".png", ".jpg")):
            arquivos_imagem.append(arquivo)

    # Limitar o número de arquivos se necessário
    if limite_arquivos and len(arquivos_imagem) > limite_arquivos:
        st.write(
            f"Limitando processamento a {limite_arquivos} arquivos de {len(arquivos_imagem)} encontrados."
        )
        arquivos_imagem = arquivos_imagem[:limite_arquivos]

    # Verificar se encontrou imagens
    if not arquivos_imagem:
        st.error(f"Nenhuma imagem encontrada na pasta {pasta_samples}!")
        return

    st.write(f"Encontradas {len(arquivos_imagem)} imagens para processar.")
    st.write(f"Usando os seguintes parâmetros:")
    st.json(params)

    # Processar cada imagem
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Criar colunas para exibir as imagens
    col1, col2 = st.columns(2)

    for i, arquivo in enumerate(arquivos_imagem):
        status_text.text(f"Processando imagem {i+1}/{len(arquivos_imagem)}: {arquivo}")

        # Caminho completo para o arquivo
        imagem_path = os.path.join(pasta_samples_path, arquivo)

        # Processar a imagem com os melhores parâmetros
        imagem_processada = processar_imagem(params, imagem_path)

        if imagem_processada is not None:
            # Nome base do arquivo (sem extensão)
            nome_base = os.path.splitext(arquivo)[0]

            # Salvar a imagem processada na pasta de resultados
            nome_arquivo_resultado = f"processado_{nome_base}.png"
            caminho_resultado = os.path.join(pasta_resultados, nome_arquivo_resultado)

            cv2.imwrite(caminho_resultado, imagem_processada)
            st.success(f"Imagem processada salva como: {caminho_resultado}")

            resultados.append(
                {
                    "arquivo_original": arquivo,
                    "arquivo_processado": nome_arquivo_resultado,
                }
            )

            # Exibir as imagens original e processada
            imagem_original = cv2.imread(imagem_path)
            imagem_original_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
            imagem_processada_rgb = cv2.cvtColor(imagem_processada, cv2.COLOR_BGR2RGB)

            with col1:
                st.image(imagem_original_rgb, caption=f"Original: {arquivo}", use_container_width=True)
            with col2:
                st.image(imagem_processada_rgb, caption=f"Processada: {nome_arquivo_resultado}", use_container_width=True)

        # Atualizar a barra de progresso
        progress_bar.progress((i + 1) / len(arquivos_imagem))

    # Limpar a barra de progresso e o texto de status
    progress_bar.empty()
    status_text.empty()

    st.success(f"Processamento concluído! {len(resultados)} imagens processadas com sucesso.")

    return resultados


# Interface principal do Streamlit
def main():
    st.sidebar.title("Opções")
    opcao = st.sidebar.radio(
        "Selecione uma opção:",
        ["Aprender com Captchas", "Processar Samples", "Fluxo Completo", "Sobre o Algoritmo"],
    )

    if opcao == "Aprender com Captchas":
        st.header("Aprendizado com Captchas")
        st.write(
            "Nesta etapa, o algoritmo genético aprende os melhores parâmetros para processar os captchas."
        )
        resultados, params_media = processar_captchas_streamlit()
        if resultados:
            # Salvar os parâmetros médios em um arquivo para uso posterior
            if params_media:
                pasta_resultados = garantir_pasta_resultados()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                params_file = os.path.join(pasta_resultados, f"params_media_{timestamp}.txt")
                with open(params_file, "w") as f:
                    f.write("Parâmetros médios:\n")
                    for param, valor in params_media.items():
                        f.write(f"  {param}: {valor}\n")
                st.success(f"Parâmetros médios salvos em: {params_file}")
            else:
                st.info("Processamento concluído. Parâmetros médios não disponíveis para um único captcha.")

    elif opcao == "Processar Samples":
        st.header("Processamento de Samples")
        st.write(
            "Nesta etapa, você pode processar as imagens da pasta 'samples' usando parâmetros predefinidos."
        )

        # Opções para os parâmetros
        st.subheader("Parâmetros para Processamento")
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Threshold", 50, 150, 100, key="samples_threshold")
            blur = st.slider("Blur", 1, 5, 3, key="samples_blur")
            dilate_size = st.slider("Dilate Size", 1, 5, 3, key="samples_dilate_size")
        with col2:
            dilate_shape = st.slider("Dilate Shape", 1, 5, 3, key="samples_dilate_shape")
            erode_size = st.slider("Erode Size", 1, 5, 3, key="samples_erode_size")
            erode_shape = st.slider("Erode Shape", 1, 5, 3, key="samples_erode_shape")

        # Criar dicionário de parâmetros
        params = {
            "threshold": threshold,
            "blur": blur,
            "dilate_size": dilate_size,
            "dilate_shape": dilate_shape,
            "erode_size": erode_size,
            "erode_shape": erode_shape,
        }

        # Opção para limitar o número de arquivos
        limite = st.number_input(
            "Limite de arquivos a processar (0 para todos)", min_value=0, value=10
        )
        if limite == 0:
            limite = None

        # Botão para iniciar o processamento
        if st.button("Iniciar Processamento de Samples", key="samples_process_btn"):
            processar_samples_streamlit(params, limite_arquivos=limite)

    elif opcao == "Fluxo Completo":
        st.header("Fluxo Completo: Aprender e Processar")
        st.write(
            "Nesta etapa, o algoritmo genético aprende com os captchas e depois aplica os parâmetros médios às imagens da pasta 'samples'."
        )

        # Configurações do algoritmo genético
        st.subheader("Configurações do Algoritmo Genético")
        col1, col2, col3 = st.columns(3)
        with col1:
            tamanho_pop = st.slider("Tamanho da População", 10, 100, 20, key="fluxo_pop")
        with col2:
            num_geracoes = st.slider("Número de Gerações", 10, 200, 50, key="fluxo_gen")
        with col3:
            taxa_mut = st.slider("Taxa de Mutação", 0.0, 1.0, 0.2, key="fluxo_mut")

        # Opção para limitar o número de arquivos
        limite = st.number_input(
            "Limite de arquivos a processar (0 para todos)", min_value=0, value=10
        )
        if limite == 0:
            limite = None

        # Botão para iniciar o fluxo completo
        if st.button("Iniciar Fluxo Completo", key="fluxo_completo_btn"):
            # Primeiro, processar os captchas para encontrar os melhores parâmetros
            resultados_captchas, params_media = processar_captchas_streamlit(
                tamanho_populacao=tamanho_pop,
                geracoes=num_geracoes,
                taxa_mutacao=taxa_mut,
                mostrar_config=False
            )

            if resultados_captchas and params_media:
                # Usar os parâmetros médios para processar as imagens da pasta samples
                processar_samples_streamlit(
                    params_media, limite_arquivos=limite
                )

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
        
        O fluxo completo do aplicativo consiste em:
        1. Aprender os melhores parâmetros usando os 5 captchas de exemplo.
        2. Calcular a média dos parâmetros encontrados.
        3. Aplicar esses parâmetros médios às imagens da pasta 'samples'.
        """
        )


# Executar a aplicação quando o script é executado diretamente
if __name__ == "__main__":
    main()