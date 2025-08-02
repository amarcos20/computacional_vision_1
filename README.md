

## Como Executar

Siga estes passos para configurar e executar o projeto no seu ambiente local.

### 1. Pré-requisitos

-   Python 3.8+
- requirements.txt

### 2. Configuração do Ambiente



```bash
git clone https://github.com/amarcos20/computacional_vision_1.git
cd computacional_vision_1
```

Crie um ambiente virtual e instale todas as dependências necessárias:

```bash
# Crie um ambiente virtual (recomendado)
python -m venv env
source env/bin/activate  # No Windows, use `env\Scripts\activate`

# Instale os pacotes
pip install -r requirements.txt
```

### 3. Executar a Pipeline de Treino (Opcional)

Se desejar treinar o modelo do zero, pode executar todo o Jupyter Notebook:
-   Abra o `notebooks/pipeline.ipynb` num ambiente Jupyter.
-   Execute todas as células pela ordem. Isto irá gerar um novo ficheiro `hand_gesture_model.h5` na pasta `/models`.

### 4. Fazer Previsões

Existem duas formas de usar o modelo treinado:

#### A) Previsão numa Imagem Estática

1.  Abra o ficheiro `predict_image.py`.
2.  Altere a variável `IMAGE_TO_PREDICT_PATH` para o caminho da imagem que quer testar.
3.  Execute o script no terminal:

```bash
python predict_image.py
```

#### B) Previsão em Tempo Real com a Webcam

Para a melhor experiência, use o script da webcam que deteta a sua mão e faz a previsão em tempo real.

Execute o script no terminal:

```bash
python predict_webcam.py
```

Pressione a tecla **'q'** para fechar a janela da webcam e terminar o programa.
