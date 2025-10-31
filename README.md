# Trabalho PAI

## Pré-requisitos

Certifique-se de ter o seguinte instalado em seu sistema:

### 1. Python 3.6 ou superior:
- **Windows**: Download from [python.org](https://python.org/downloads/)
- **macOS**:
  ```bash
  # Usando Homebrew
  brew install python
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip
  ```

### 2. pip (Gerenciador de pacotes Python):
- Geralmente vem junto com Python 3.4+
- Verifique a instalação:
  ```bash
  python -m pip --version
  ```
- Se não estiver instalado:
  ```bash
  python -m ensurepip --upgrade
  ```

## Inicialização

Siga estas etapas para executar o aplicativo:

### 1. Criar e ativar um ambiente virtual:
- **macOS/Linux**:
```bash
python -m venv .venv
source .venv/bin/activate
```
- **Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

*Você deverá ver `(.venv)` no prompt do terminal indicando que o ambiente virtual está ativo.*

### 2. Instalar dependências

Com seu ambiente virtual ativado, instale todos os pacotes necessários:

```bash
pip install -r requirements.txt
```

### 3. Execute o aplicativo

```bash
python app.py
```

Você deverá ver uma saída semelhante a:
```
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
```