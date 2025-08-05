import random
import numpy as np
import os
import cv2
from datetime import datetime
import pandas as pd
from processamento_imagem import (
    processar_imagem,
    calcular_similaridade,
    garantir_pasta_resultados,
)


def criar_individuo():
    """
    Cria um indivíduo aleatório para a população inicial.

    Returns:
        Dicionário com os parâmetros do indivíduo
    """
    return {
        "threshold": random.randint(50, 150),
        "blur": random.randint(1, 5),
        "dilate_size": random.randint(1, 5),
        "dilate_shape": random.randint(1, 5),
        "erode_size": random.randint(1, 5),
        "erode_shape": random.randint(1, 5),
    }


def criar_populacao(tamanho):
    """
    Cria uma população inicial de indivíduos.

    Args:
        tamanho: Número de indivíduos na população

    Returns:
        Lista de indivíduos (dicionários de parâmetros)
    """
    return [criar_individuo() for _ in range(tamanho)]


def avaliar_individuo(individuo, imagem_path, imagem_alvo_path):
    """
    Avalia a aptidão de um indivíduo processando a imagem e comparando com a imagem alvo.

    Args:
        individuo: Dicionário com os parâmetros do indivíduo
        imagem_path: Caminho para a imagem a ser processada
        imagem_alvo_path: Caminho para a imagem alvo

    Returns:
        Valor de aptidão (similaridade) entre 0 e 1
    """
    # Processar a imagem com os parâmetros do indivíduo
    imagem_processada = processar_imagem(individuo, imagem_path)
    if imagem_processada is None:
        return 0

    # Carregar a imagem alvo
    imagem_alvo = cv2.imread(imagem_alvo_path)
    if imagem_alvo is None:
        print(f"Erro ao carregar a imagem alvo: {imagem_alvo_path}")
        return 0

    # Calcular a similaridade
    similaridade = calcular_similaridade(imagem_processada, imagem_alvo)
    return similaridade


def selecionar_pais(populacao, aptidoes):
    """
    Seleciona dois pais da população usando o método da roleta.

    Args:
        populacao: Lista de indivíduos
        aptidoes: Lista de valores de aptidão correspondentes

    Returns:
        Tupla com dois indivíduos selecionados como pais
    """
    # Calcular a soma total das aptidões
    soma_aptidoes = sum(aptidoes)
    if soma_aptidoes == 0:
        # Se todas as aptidões forem zero, selecionar aleatoriamente
        return random.sample(populacao, 2)

    # Normalizar as aptidões para criar probabilidades
    probabilidades = [apt / soma_aptidoes for apt in aptidoes]

    # Selecionar dois pais usando o método da roleta
    pais = []
    for _ in range(2):
        # Gerar um número aleatório entre 0 e 1
        r = random.random()
        # Percorrer a população e selecionar um indivíduo
        soma_prob = 0
        for i, prob in enumerate(probabilidades):
            soma_prob += prob
            if r <= soma_prob:
                pais.append(populacao[i])
                break
        # Se não selecionou ninguém (pode acontecer devido a erros de arredondamento)
        if len(pais) <= _:
            pais.append(random.choice(populacao))

    return pais[0], pais[1]


def cruzamento(pai1, pai2):
    """
    Realiza o cruzamento entre dois pais para gerar um filho.

    Args:
        pai1: Primeiro pai (dicionário de parâmetros)
        pai2: Segundo pai (dicionário de parâmetros)

    Returns:
        Novo indivíduo (dicionário de parâmetros) gerado pelo cruzamento
    """
    filho = {}
    for param in pai1.keys():
        # 50% de chance de herdar de cada pai
        if random.random() < 0.5:
            filho[param] = pai1[param]
        else:
            filho[param] = pai2[param]
    return filho


def mutacao(individuo, taxa_mutacao):
    """
    Aplica mutação a um indivíduo com uma certa probabilidade.

    Args:
        individuo: Indivíduo a ser mutado (dicionário de parâmetros)
        taxa_mutacao: Probabilidade de ocorrer mutação em cada parâmetro

    Returns:
        Indivíduo após a mutação
    """
    for param in individuo.keys():
        # Verificar se ocorre mutação neste parâmetro
        if random.random() < taxa_mutacao:
            # Aplicar mutação de acordo com o tipo de parâmetro
            if param == "threshold":
                individuo[param] = random.randint(50, 150)
            else:
                individuo[param] = random.randint(1, 5)
    return individuo


def executar_algoritmo_genetico(
    imagem_path,
    imagem_alvo_path,
    tamanho_populacao=20,
    geracoes=50,
    taxa_mutacao=0.2,
    callback=None,
):
    """
    Executa o algoritmo genético para encontrar os melhores parâmetros de processamento.

    Args:
        imagem_path: Caminho para a imagem a ser processada
        imagem_alvo_path: Caminho para a imagem alvo
        tamanho_populacao: Tamanho da população
        geracoes: Número de gerações
        taxa_mutacao: Taxa de mutação
        callback: Função de callback para atualizar a interface (opcional)

    Returns:
        Tupla com o melhor indivíduo e seu valor de aptidão
    """
    # Criar a população inicial
    populacao = criar_populacao(tamanho_populacao)

    # Melhor indivíduo global
    melhor_global = None
    melhor_aptidao_global = 0

    # Histórico de aptidões e parâmetros
    historico_aptidoes = []
    historico_parametros = {}
    for param in populacao[0].keys():
        historico_parametros[param] = []

    # Loop principal do algoritmo genético
    for geracao in range(geracoes):
        # Avaliar cada indivíduo da população
        aptidoes = []
        for individuo in populacao:
            aptidao = avaliar_individuo(individuo, imagem_path, imagem_alvo_path)
            aptidoes.append(aptidao)

        # Encontrar o melhor indivíduo desta geração
        melhor_indice = aptidoes.index(max(aptidoes))
        melhor_individuo = populacao[melhor_indice]
        melhor_aptidao = aptidoes[melhor_indice]

        # Atualizar o melhor global se necessário
        if melhor_aptidao > melhor_aptidao_global:
            melhor_global = melhor_individuo.copy()
            melhor_aptidao_global = melhor_aptidao

        # Registrar histórico
        historico_aptidoes.append(melhor_aptidao)
        for param, valor in melhor_individuo.items():
            historico_parametros[param].append(valor)

        # Chamar a função de callback, se fornecida
        if callback:
            continuar = callback(
                geracao=geracao,
                geracoes=geracoes,
                melhor_individuo=melhor_individuo,
                melhor_aptidao=melhor_aptidao,
                melhor_global=melhor_global,
                melhor_aptidao_global=melhor_aptidao_global,
                historico_aptidoes=historico_aptidoes,
                historico_parametros=historico_parametros,
            )
            if not continuar:
                break

        # Criar a nova população
        nova_populacao = []

        # Elitismo: manter o melhor indivíduo
        nova_populacao.append(melhor_individuo.copy())

        # Gerar o resto da população
        while len(nova_populacao) < tamanho_populacao:
            # Selecionar pais
            pai1, pai2 = selecionar_pais(populacao, aptidoes)

            # Cruzamento
            filho = cruzamento(pai1, pai2)

            # Mutação
            filho = mutacao(filho, taxa_mutacao)

            # Adicionar à nova população
            nova_populacao.append(filho)

        # Substituir a população antiga pela nova
        populacao = nova_populacao

    return (
        melhor_global,
        melhor_aptidao_global,
        historico_aptidoes,
        historico_parametros,
    )


def salvar_resultados(
    captcha_nome, melhor_individuo, melhor_aptidao, imagem_original, imagem_processada
):
    """
    Salva os resultados do processamento de um captcha.

    Args:
        captcha_nome: Nome do arquivo de captcha
        melhor_individuo: Melhores parâmetros encontrados
        melhor_aptidao: Valor de aptidão do melhor indivíduo
        imagem_original: Imagem original do captcha
        imagem_processada: Imagem processada com os melhores parâmetros

    Returns:
        Dicionário com informações sobre o resultado
    """
    # Garantir que a pasta de resultados exista
    pasta_resultados = garantir_pasta_resultados()

    # Criar um timestamp para os nomes dos arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nome base para os arquivos (sem extensão)
    nome_base = os.path.splitext(captcha_nome)[0]

    # Salvar a imagem processada
    nome_arquivo_processado = f"{nome_base}_processado_{timestamp}.png"
    caminho_processado = os.path.join(pasta_resultados, nome_arquivo_processado)
    cv2.imwrite(caminho_processado, imagem_processada)

    # Salvar os parâmetros em um arquivo de texto
    nome_arquivo_params = f"{nome_base}_params_{timestamp}.txt"
    caminho_params = os.path.join(pasta_resultados, nome_arquivo_params)
    with open(caminho_params, "w") as f:
        f.write(f"Parâmetros para {captcha_nome}:\n")
        f.write(f"Aptidão: {float(melhor_aptidao):.4f}\n")
        for param, valor in melhor_individuo.items():
            f.write(f"  {param}: {valor}\n")

    # Retornar informações sobre o resultado
    return {
        "captcha": captcha_nome,
        "aptidao": melhor_aptidao,
        "parametros": melhor_individuo,
        "imagem_processada": nome_arquivo_processado,
        "arquivo_parametros": nome_arquivo_params,
    }


def calcular_media_parametros(resultados):
    """
    Calcula a média dos parâmetros de vários resultados.

    Args:
        resultados: Lista de dicionários com resultados

    Returns:
        Dicionário com os parâmetros médios
    """
    if not resultados:
        return None

    # Inicializar dicionário para somar os parâmetros
    soma_params = {}
    for param in resultados[0]["parametros"].keys():
        soma_params[param] = 0

    # Somar os parâmetros de todos os resultados
    for resultado in resultados:
        for param, valor in resultado["parametros"].items():
            soma_params[param] += valor

    # Calcular a média
    media_params = {}
    for param, soma in soma_params.items():
        media_params[param] = int(round(soma / len(resultados)))

    return media_params
