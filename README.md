# 🧬 Algoritmo Genético - Otimização Evolutiva de Captcha

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Características e Funcionalidades](#-características-e-funcionalidades)
- [Interface de Visualização com Streamlit](#-interface-de-visualização-com-streamlit)
- [Arquitetura](#-arquitetura)
- [Instalação e Execução do Projeto](#-instalação-e-execução-do-projeto)
- [Configuração](#-configuração)
- [Equipe](#-equipe)
- [Licença](#-licença)

## 🎯 Sobre o Projeto

Este projeto implementa um sistema de processamento de imagens de captcha utilizando um **Algoritmo Genético (AG)** robusto e configurável para otimizar parâmetros de processamento de imagem. Inspirado nos princípios da evolução natural, o algoritmo utiliza operadores genéticos como seleção, crossover e mutação para evoluir soluções ao longo de gerações sucessivas. O objetivo é remover linhas e ruídos de imagens de captcha para melhorar a legibilidade dos caracteres, facilitando sua posterior leitura por sistemas OCR ou humanos.

## ✨ Características e Funcionalidades

- **🔧 Otimização de Parâmetros**: Encontra automaticamente os melhores parâmetros para processamento de imagens (threshold, blur, dilatação e erosão)
- **🔄 Threshold (limiarização)**: Converte a imagem para preto e branco com base em um valor limite. Para que serve? Separar objetos do fundo (por ex: detectar texto ou formas).
- **📝 Blur (borramento/desfoque)**: Aplica um desfoque para suavizar a imagem. Para que serve? Reduz ruído (pixels aleatórios), antes de detectar contornos ou aplicar threshold.
- **👁️ Dilate (dilatação)**: Aumenta as regiões brancas. Para que serve? Preencher buracos em objetos ou unir partes desconectadas.
- **🚀 Erode (erosão)**: Reduz as regiões brancas. Para que serve? Remover pequenos ruídos ou separar objetos grudados.
- **🎲 Avaliação de Aptidão**: Calcula a similaridade entre a imagem processada e uma imagem alvo ideal
- **📈 Evolução da População**: Implementa seleção, cruzamento e mutação para evoluir os parâmetros ao longo das gerações
- **📊 Processamento em Lote**: Processa múltiplos captchas sequencialmente e salva os resultados

## 🖥️ Interface de Visualização com Streamlit
- **Visualização em Tempo Real**: Acompanha a evolução do algoritmo genético geração por geração
- **Gráficos de Desempenho**: Exibe gráficos de aptidão ao longo das gerações
- **Visualização de Imagens**: Mostra a imagem original, processada e alvo lado a lado
- **Configuração Interativa**: Permite ajustar parâmetros do algoritmo genético

## 🏗️ Arquitetura

```
genetic-algorithm/
├── imgs/
│   ├── captcha1.png
│   ├── captcha1_target.png
│   ├── ...
├── samples/
│   ├── 2b827.png
│   ├── 3b4we.png
│   ├── ...
├── algoritmo_genetico_processador_streamlit.py
├── obter_captchas_kaggle.py
├── requirements.txt
└── README.md
└── LICENSE
```

## 🚀 Instalação e Execução do Projeto

### Pré-requisitos

- Python 3.6 ou superior
- Pip
- OpenCV Python (cv2)
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Kagglehub

### Instalação via pip

```bash
# Clone o repositório
git clone https://github.com/RafaelGuido/TC_FASE2
cd genetic-algorithm-processing-captcha

# Instale as dependências
pip install -r requirements.txt

# Para iniciar a interface Streamlit e visualizar a evolução do algoritmo genético:
python -m streamlit run algoritmo_genetico_streamlit.py
```

## ⚙️ Configuração

### Fluxo de Trabalho

1. **Fase de Treinamento**:
   - O algoritmo genético processa os pares de captcha/target na pasta `imgs`
   - Para cada par, evolui uma população de parâmetros por várias gerações
   - Encontra os melhores parâmetros que transformam o captcha original em algo próximo ao target
   - Salva os melhores parâmetros e imagens processadas

2. **Aplicação em Novos Captchas**:
   - Utiliza os melhores parâmetros encontrados (ou uma média deles)
   - Aplica esses parâmetros às imagens na pasta `samples`
   - Salva as imagens processadas na pasta `resultados`

3. **Visualização e Análise**:
   - A interface Streamlit permite visualizar todo o processo de evolução
   - Mostra gráficos de aptidão, parâmetros e imagens processadas
   - Facilita a compreensão e ajuste do algoritmo genético

### Parâmetros Otimizados
O algoritmo genético otimiza os seguintes parâmetros de processamento de imagem:

- `threshold`: Valor de limiarização (50–150)
- `blur`: Tamanho do kernel de desfoque (1–5)
- `dilate_size`: Tamanho do kernel de dilatação (1–5)
- `dilate_shape`: Forma do kernel de dilatação (1–5)
- `erode_size`: Tamanho do kernel de erosão (1–5)
- `erode_shape`: Forma do kernel de erosão (1–5)

## 👥 Equipe

Este projeto foi desenvolvido por:

- **[Guilherme Santana](https://www.linkedin.com/in/guilherme-santana-04360917a/)**

- **[Franklin Araujo](https://www.linkedin.com/in/franklinarauj/)**

- **[Rafael Toccolini](https://www.linkedin.com/in/rafaeltoccolini/)**

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.