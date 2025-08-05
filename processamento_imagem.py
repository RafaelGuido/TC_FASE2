import cv2
import numpy as np
import os


def calcular_similaridade(img1, img2):
    """
    Calcula a similaridade entre duas imagens usando o método de correlação normalizada.
    
    Args:
        img1: Primeira imagem para comparação
        img2: Segunda imagem para comparação
        
    Returns:
        Valor de similaridade entre -1 e 1, onde 1 indica imagens idênticas
    """
    # Garantir que as imagens têm o mesmo tamanho
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calcular a similaridade usando o método de correlação normalizada
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]


def processar_imagem(params, imagem_path):
    """
    Processa uma imagem com os parâmetros fornecidos.
    
    Args:
        params: Dicionário com os parâmetros de processamento
        imagem_path: Caminho para a imagem a ser processada
        
    Returns:
        Imagem processada ou None se ocorrer um erro
    """
    try:
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
            print(f"Erro ao carregar a imagem: {imagem_path}")
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
    
    except Exception as e:
        print(f"Erro ao processar a imagem {imagem_path}: {str(e)}")
        return None


def garantir_pasta_resultados(pasta="resultados"):
    """
    Garante que a pasta de resultados exista.
    
    Args:
        pasta: Nome da pasta de resultados (padrão: "resultados")
        
    Returns:
        Caminho para a pasta de resultados
    """
    # Obter caminho absoluto da pasta de resultados
    pasta_path = os.path.join(os.getcwd(), pasta)

    # Criar a pasta se não existir
    if not os.path.exists(pasta_path):
        os.makedirs(pasta_path)
        print(f"Pasta '{pasta}' criada com sucesso.")

    return pasta_path