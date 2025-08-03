 # Processamento de Captchas com Algoritmo Genético

## Visão Geral
Este projeto implementa um sistema de processamento de imagens de captcha utilizando algoritmos genéticos para otimizar parâmetros de processamento de imagem. O objetivo é remover linhas e ruídos de imagens de captcha para melhorar a legibilidade dos caracteres, facilitando sua posterior leitura por sistemas OCR ou humanos.

## Funcionalidades

### Algoritmo Genético
- **Otimização de Parâmetros**: Encontra automaticamente os melhores parâmetros para processamento de imagens (threshold, blur, dilatação e erosão)
1. Threshold (limiarização)
Converte a imagem para preto e branco com base em um valor limite.
Para que serve? Separar objetos do fundo (por ex: detectar texto ou formas).

2. Blur (borramento/desfoque)
Aplica um desfoque para suavizar a imagem.
Para que serve? Reduz ruído (pixels aleatórios), antes de detectar contornos ou aplicar threshold.

3. Dilate (dilatação)
Aumenta as regiões brancas.
Para que serve? Preencher buracos em objetos ou unir partes desconectadas.

4. Erode (erosão)
Reduz as regiões brancas.
Para que serve? Remover pequenos ruídos ou separar objetos grudados.

- **Avaliação de Aptidão**: Calcula a similaridade entre a imagem processada e uma imagem alvo ideal
- **Evolução da População**: Implementa seleção, cruzamento e mutação para evoluir os parâmetros ao longo das gerações
- **Processamento em Lote**: Processa múltiplos captchas sequencialmente e salva os resultados

### Interface de Visualização com Streamlit
- **Visualização em Tempo Real**: Acompanha a evolução do algoritmo genético geração por geração
- **Gráficos de Desempenho**: Exibe gráficos de aptidão ao longo das gerações
- **Visualização de Imagens**: Mostra a imagem original, processada e alvo lado a lado
- **Configuração Interativa**: Permite ajustar parâmetros do algoritmo genético

## Requisitos
- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Instalação
1. Clone o repositório ou baixe os arquivos
2. Instale as dependências necessárias:
   ```bash
   pip install opencv-python numpy pandas matplotlib streamlit
   ```

## Uso

### Executar o Algoritmo Genético
Para executar o algoritmo genético e processar os captchas:
```bash
python asa_genetico.py
```
Este comando processará todos os pares de captcha/target na pasta `imgs` e salvará os resultados.

### Visualizar a Evolução com Streamlit
Para iniciar a interface Streamlit e visualizar a evolução do algoritmo genético:
```bash
python -m streamlit run asa_genetico_streamlit.py
```

## Estrutura de Arquivos
- `asa.py`: Script básico de processamento de imagens com OpenCV  
- `asa_genetico.py`: Implementação principal do algoritmo genético para processamento de captchas  
- `asa_genetico_streamlit.py`: Interface Streamlit para visualização da evolução do algoritmo genético  
- `imgs/`: Pasta contendo os pares de imagens de captcha e seus targets (ex: `captcha1.png`, `captcha1_target.png`)  
- `samples/`: Pasta contendo imagens de captcha para processamento em lote  
- `resultados/`: Pasta onde são salvos os resultados do processamento  

## Fluxo de Trabalho

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

## Parâmetros Otimizados
O algoritmo genético otimiza os seguintes parâmetros de processamento de imagem:

- `threshold`: Valor de limiarização (50–150)  
- `blur`: Tamanho do kernel de desfoque (1–5)  
- `dilate_size`: Tamanho do kernel de dilatação (1–5)  
- `dilate_shape`: Forma do kernel de dilatação (1–5)  
- `erode_size`: Tamanho do kernel de erosão (1–5)  
- `erode_shape`: Forma do kernel de erosão (1–5)  

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* ou enviar *pull requests* com melhorias.