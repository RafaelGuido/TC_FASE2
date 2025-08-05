import streamlit as st
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Importar funções dos outros módulos
from processamento_imagem import (
    processar_imagem,
    calcular_similaridade,
    garantir_pasta_resultados,
)
from algoritmo_genetico import (
    executar_algoritmo_genetico,
    salvar_resultados,
    calcular_media_parametros,
)


def processar_captchas_streamlit(
    tamanho_populacao=20, geracoes=50, taxa_mutacao=0.2, mostrar_config=True
):
    """
    Processa os captchas usando o algoritmo genético e exibe os resultados no Streamlit.

    Args:
        tamanho_populacao: Tamanho da população para o algoritmo genético
        geracoes: Número de gerações para o algoritmo genético
        taxa_mutacao: Taxa de mutação para o algoritmo genético
        mostrar_config: Se True, exibe as configurações do algoritmo genético

    Returns:
        Tupla com a lista de resultados e os parâmetros médios
    """
    # Configurações do algoritmo genético
    if mostrar_config:
        st.subheader("Configurações do Algoritmo Genético")
        col1, col2, col3 = st.columns(3)
        with col1:
            tamanho_populacao = st.slider(
                "Tamanho da População", 10, 100, tamanho_populacao
            )
        with col2:
            geracoes = st.slider("Número de Gerações", 10, 200, geracoes)
        with col3:
            taxa_mutacao = st.slider("Taxa de Mutação", 0.0, 1.0, taxa_mutacao)

        # Opção para processar todos os captchas ou apenas um
        st.subheader("Seleção de Captchas")
        opcao_captcha = st.radio(
            "Selecione uma opção:",
            ["Processar todos os captchas", "Processar apenas um captcha"],
        )

        # Pasta de imagens
        pasta_imgs = os.path.join(os.getcwd(), "imgs")
        if not os.path.exists(pasta_imgs):
            st.error(f"Pasta de imagens não encontrada: {pasta_imgs}")
            return None, None

        # Listar os arquivos de captcha disponíveis
        captchas = []
        for arquivo in os.listdir(pasta_imgs):
            if (
                arquivo.lower().endswith((".png", ".jpg"))
                and "target" not in arquivo.lower()
            ):
                captchas.append(arquivo)

        if not captchas:
            st.error(f"Nenhum arquivo de captcha encontrado na pasta {pasta_imgs}!")
            return None, None

        # Se a opção for processar apenas um captcha
        if opcao_captcha == "Processar apenas um captcha":
            captcha_selecionado = st.selectbox("Selecione um captcha:", captchas)
            captchas = [captcha_selecionado]

        # Botão para iniciar o processamento
        iniciar = st.button("Iniciar Processamento")
        if not iniciar:
            return None, None
    else:
        # Se não mostrar configuração, processar todos os captchas automaticamente
        pasta_imgs = os.path.join(os.getcwd(), "imgs")
        if not os.path.exists(pasta_imgs):
            return None, None

        captchas = []
        for arquivo in os.listdir(pasta_imgs):
            if (
                arquivo.lower().endswith((".png", ".jpg"))
                and "target" not in arquivo.lower()
            ):
                captchas.append(arquivo)

        if not captchas:
            return None, None

    # Processar cada captcha
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, captcha in enumerate(captchas):
        st.subheader(f"Processando captcha: {captcha}")

        # Caminhos para as imagens
        captcha_path = os.path.join(pasta_imgs, captcha)
        nome_base = os.path.splitext(captcha)[0]
        target_path = os.path.join(pasta_imgs, f"{nome_base}_target.png")

        # Verificar se a imagem alvo existe
        if not os.path.exists(target_path):
            st.error(f"Imagem alvo não encontrada: {target_path}")
            continue

        # Exibir as imagens original e alvo
        col1, col2 = st.columns(2)
        with col1:
            captcha_img = cv2.imread(captcha_path)
            captcha_img_rgb = cv2.cvtColor(captcha_img, cv2.COLOR_BGR2RGB)
            st.image(
                captcha_img_rgb, caption=f"Captcha: {captcha}", use_container_width=True
            )

        with col2:
            target_img = cv2.imread(target_path)
            target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            st.image(
                target_img_rgb,
                caption=f"Alvo: {nome_base}_target.png",
                use_container_width=True,
            )

        # Criar containers para exibir informações durante o processamento
        info_container = st.container()  # Será usado apenas no final
        fitness_container = st.empty()  # Container para exibir o valor atual da aptidão
        chart_container = st.empty()  # Usamos st.empty() para atualizar o mesmo gráfico
        image_container = st.empty()  # Usamos st.empty() para atualizar a mesma imagem

        # Função de callback para atualizar a interface durante o processamento
        def update_ui(
            geracao,
            geracoes,
            melhor_individuo,
            melhor_aptidao,
            melhor_global,
            melhor_aptidao_global,
            historico_aptidoes,
            historico_parametros,
        ):
            # Atualizar texto de status
            status_text.text(
                f"Processando captcha {i+1}/{len(captchas)}: {captcha} - Geração {geracao+1}/{geracoes}"
            )

            # Atualizar barra de progresso
            progress = (i / len(captchas)) + ((geracao + 1) / geracoes) / len(captchas)
            progress_bar.progress(progress)

            # Exibir o valor atual da aptidão em um container separado
            fitness_html = f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="flex: 1;">
                    <h3>Aptidão Atual: <strong>{float(melhor_aptidao):.4f}</strong></h3>
                </div>
            </div>
            """

            # Adicionar informação sobre a melhor aptidão global se disponível
            if melhor_aptidao_global > 0:
                fitness_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="flex: 1;">
                        <h3>Melhor Aptidão Global: <strong>{float(melhor_aptidao_global):.4f}</strong></h3>
                    </div>
                </div>
                """

            # Atualizar gráficos usando o container vazio para substituir o anterior
            # Criar figura com dois subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Gráfico de aptidão com valor atual e melhor global
            ax1.plot(historico_aptidoes)
            ax1.set_title(
                f"Evolução da Aptidão (Atual: {float(melhor_aptidao):.4f} - Melhor: {float(melhor_aptidao_global):.4f})"
            )
            ax1.set_xlabel("Geração")
            ax1.set_ylabel("Aptidão")
            ax1.grid(True)

            # Adicionar texto com o valor atual da aptidão e melhor global
            ax1.text(
                0.02,
                0.95,
                f"Aptidão atual: {float(melhor_aptidao):.4f}\nMelhor aptidão: {float(melhor_aptidao_global):.4f}",
                transform=ax1.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # Gráfico de parâmetros
            for param, valores in historico_parametros.items():
                ax2.plot(valores, label=param)
            ax2.set_title("Evolução dos Parâmetros")
            ax2.set_xlabel("Geração")
            ax2.set_ylabel("Valor")
            ax2.legend()
            ax2.grid(True)

            # Adicionar texto com os valores atuais dos parâmetros
            param_text = "\n".join(
                [f"{param}: {melhor_individuo[param]}" for param in melhor_individuo]
            )
            ax2.text(
                0.02,
                0.95,
                param_text,
                transform=ax2.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            # Ajustar layout e exibir
            plt.tight_layout()
            chart_container.pyplot(fig)
            plt.close(fig)

            # Processar a imagem com os melhores parâmetros e exibir apenas a cada 5 gerações ou na última
            if geracao % 5 == 0 or geracao == geracoes - 1:
                imagem_processada = processar_imagem(melhor_global, captcha_path)
                if imagem_processada is not None:
                    imagem_processada_rgb = cv2.cvtColor(
                        imagem_processada, cv2.COLOR_BGR2RGB
                    )
                    image_container.image(
                        imagem_processada_rgb,
                        caption=f"Imagem Processada (Geração {geracao+1})",
                        use_container_width=True,
                    )

            # Continuar o processamento
            return True

        # Executar o algoritmo genético
        melhor_individuo, melhor_aptidao, historico_aptidoes, historico_parametros = (
            executar_algoritmo_genetico(
                captcha_path,
                target_path,
                tamanho_populacao=tamanho_populacao,
                geracoes=geracoes,
                taxa_mutacao=taxa_mutacao,
                callback=update_ui,
            )
        )

        # Processar a imagem com os melhores parâmetros
        imagem_original = cv2.imread(captcha_path)
        imagem_processada = processar_imagem(melhor_individuo, captcha_path)

        if imagem_processada is not None:
            # Salvar os resultados
            resultado = salvar_resultados(
                captcha,
                melhor_individuo,
                melhor_aptidao,
                imagem_original,
                imagem_processada,
            )
            resultados.append(resultado)

            # Exibir o resultado final
            st.success(f"Processamento concluído para {captcha}!")

            # Exibir informações detalhadas sobre o resultado
            info_container_final = st.container()
            with info_container_final:

                # Exibir os melhores parâmetros em formato de tabela para melhor visualização
                st.write("**Melhores parâmetros:**")

                # Criar um DataFrame para exibir os parâmetros de forma mais organizada
                params_df = pd.DataFrame(
                    {
                        "Parâmetro": list(melhor_individuo.keys()),
                        "Valor": list(melhor_individuo.values()),
                    }
                )
                st.table(params_df)

                # Também exibir em formato JSON para compatibilidade
                with st.expander("Ver parâmetros em formato JSON"):
                    st.json(melhor_individuo)

                # Exibir informações sobre os arquivos salvos
                st.write(
                    f"**Imagem processada salva como:** {resultado['imagem_processada']}"
                )
                st.write(f"**Parâmetros salvos em:** {resultado['arquivo_parametros']}")

        else:
            st.error(f"Erro ao processar a imagem {captcha}!")

    # Limpar a barra de progresso e o texto de status
    progress_bar.empty()
    status_text.empty()

    # Calcular a média dos parâmetros se houver mais de um resultado
    params_media = None
    if len(resultados) > 1:
        params_media = calcular_media_parametros(resultados)

    # Exibir resumo dos resultados
    if resultados:
        st.subheader("Resumo dos Resultados")

        # Criar um DataFrame com os resultados
        df_resultados = pd.DataFrame(
            [
                {
                    "Captcha": r["captcha"],
                    "Aptidão": r["aptidao"],
                    **{f"Param_{k}": v for k, v in r["parametros"].items()},
                }
                for r in resultados
            ]
        )

        # Exibir o DataFrame
        st.dataframe(df_resultados)

    return resultados, params_media


def processar_samples_streamlit(params, limite_arquivos=None):
    """
    Processa as imagens da pasta 'samples' usando os parâmetros fornecidos.

    Args:
        params: Dicionário com os parâmetros de processamento
        limite_arquivos: Limite de arquivos a processar (None para processar todos)

    Returns:
        Lista de resultados do processamento
    """
    # Verificar se os parâmetros foram fornecidos
    if not params:
        st.error("Parâmetros não fornecidos!")
        return

    # Pasta de samples
    pasta_samples = "samples"
    pasta_samples_path = os.path.join(os.getcwd(), pasta_samples)
    if not os.path.exists(pasta_samples_path):
        st.error(f"Pasta de samples não encontrada: {pasta_samples_path}")
        return

    # Garantir que a pasta de resultados exista
    pasta_resultados = garantir_pasta_resultados()

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
                st.image(
                    imagem_original_rgb,
                    caption=f"Original: {arquivo}",
                    use_container_width=True,
                )
            with col2:
                st.image(
                    imagem_processada_rgb,
                    caption=f"Processada: {nome_arquivo_resultado}",
                    use_container_width=True,
                )

        # Atualizar a barra de progresso
        progress_bar.progress((i + 1) / len(arquivos_imagem))

    # Limpar a barra de progresso e o texto de status
    progress_bar.empty()
    status_text.empty()

    st.success(
        f"Processamento concluído! {len(resultados)} imagens processadas com sucesso."
    )

    return resultados


# Interface principal do Streamlit
def main():
    st.sidebar.title("Opções")
    opcao = st.sidebar.radio(
        "Selecione uma opção:",
        [
            "Home",
            "Aprender com Captchas",
            "Processar Samples",
            "Fluxo Completo",
            "Sobre o Algoritmo",
        ],
    )

    if opcao == "Home":
        st.title("🧬 Pré-Processador de Captchas com Algoritmo Genético")

        st.markdown(
            """
        ## Bem-vindo ao Sistema de Processamento de Captchas!
        
        Este aplicativo utiliza algoritmos genéticos para otimizar o processamento de imagens de captcha.
        
        ### Funcionalidades Principais:
        
        - **Aprender com Captchas**: Treina o algoritmo genético com imagens de captcha para encontrar os melhores parâmetros de processamento.
        - **Processar Samples**: Aplica parâmetros predefinidos às imagens da pasta 'samples'.
        - **Fluxo Completo**: Executa o processo de aprendizado e aplicação em um único fluxo.
        - **Sobre o Algoritmo**: Informações detalhadas sobre o funcionamento do algoritmo genético.
        
        ### Como Começar:
        
        1. Selecione uma das opções no menu lateral à esquerda
        2. Siga as instruções específicas de cada módulo
        3. Visualize os resultados e ajuste os parâmetros conforme necessário
        
        Desenvolvido para o desafio TC_FASE2 da FIAP.
        """
        )

        # Exibir estatísticas do projeto
        st.subheader("Estatísticas do Projeto")
        col1, col2, col3 = st.columns(3)

        # Contar arquivos nas pastas
        pasta_imgs = os.path.join(os.getcwd(), "imgs")
        pasta_samples = os.path.join(os.getcwd(), "samples")
        pasta_resultados = os.path.join(os.getcwd(), "resultados")

        with col1:
            num_captchas = 0
            if os.path.exists(pasta_imgs):
                num_captchas = len(
                    [
                        f
                        for f in os.listdir(pasta_imgs)
                        if f.lower().endswith((".png", ".jpg"))
                        and "target" not in f.lower()
                    ]
                )
            st.metric("Captchas Disponíveis", num_captchas)

        with col2:
            num_samples = 0
            if os.path.exists(pasta_samples):
                num_samples = len(
                    [
                        f
                        for f in os.listdir(pasta_samples)
                        if f.lower().endswith((".png", ".jpg"))
                    ]
                )
            st.metric("Samples para Processar", num_samples)

        with col3:
            num_resultados = 0
            if os.path.exists(pasta_resultados):
                num_resultados = len(
                    [
                        f
                        for f in os.listdir(pasta_resultados)
                        if f.startswith("processado_")
                    ]
                )
            st.metric("Resultados Gerados", num_resultados)

    elif opcao == "Aprender com Captchas":
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
                params_file = os.path.join(
                    pasta_resultados, f"params_media_{timestamp}.txt"
                )
                with open(params_file, "w") as f:
                    f.write("Parâmetros médios:\n")
                    for param, valor in params_media.items():
                        f.write(f"  {param}: {valor}\n")
                st.success(f"Parâmetros médios salvos em: {params_file}")
            else:
                st.info(
                    "Processamento concluído. Parâmetros médios não disponíveis para um único captcha."
                )

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
            dilate_shape = st.slider(
                "Dilate Shape", 1, 5, 3, key="samples_dilate_shape"
            )
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
            tamanho_pop = st.slider(
                "Tamanho da População", 10, 100, 20, key="fluxo_pop"
            )
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
            with st.spinner(
                "Processando captchas para encontrar os melhores parâmetros..."
            ):
                resultados_captchas, params_media = processar_captchas_streamlit(
                    tamanho_populacao=tamanho_pop,
                    geracoes=num_geracoes,
                    taxa_mutacao=taxa_mut,
                    mostrar_config=False,
                )

                if resultados_captchas and params_media:
                    # Usar os parâmetros médios para processar as imagens da pasta samples
                    st.success(
                        "Parâmetros ótimos encontrados! Processando imagens da pasta samples..."
                    )
                    processar_samples_streamlit(params_media, limite_arquivos=limite)
                else:
                    st.error(
                        "Não foi possível obter parâmetros ótimos. Verifique se os captchas e imagens alvo existem."
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
