# AR - Aprendizagem por Reforço

Repositório da disciplina com exercícios.

## Requisitos

- Python 3.12+ (o projeto exige `>=3.12`)
- [uv](https://docs.astral.sh/uv/) instalado

### Instalar o `uv`

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verificar instalação:

```bash
uv --version
```

## Setup do projeto

No diretório raiz (`ar/`):

```bash
uv sync
```

Esse comando:
- cria automaticamente `.venv/` (se não existir)
- instala todas as dependências definidas no `pyproject.toml`

Dependências principais deste repositório:
- `numpy`
- `matplotlib`
- `ipykernel`
- `torch`

## Como usar

### 1) Executar scripts Python

Sem ativar ambiente manualmente:

```bash
uv run python <file>
```

### 2) Trabalhar com notebooks

Fluxo recomendado no VS Code:

1. Rode `uv sync` na raiz do projeto.
2. Abra o notebook (`.ipynb`).
3. Selecione o kernel Python da `.venv` do projeto.

## Comandos úteis com `uv`

Adicionar nova dependência:

```bash
uv add <pacote>
```

Exemplo:

```bash
uv add matplotlib
```

Atualizar lock/dependências do ambiente novamente:

```bash
uv sync
```

## Estrutura do projeto

```text
ar/
├─ pyproject.toml
├─ README.md
├─ aulas/
│  ├─ aula_2026-02-10/
│  ├─ aula_2026-02-24/
│  ├─ aula_2026-03-03/
│  ├─ aula_2026-03-10/
│  ├─ aula_2026-03-17/
│  ├─ aula_2026-03-24/
│  ├─ aula_2026-04-07/
│  ├─ aula_2026-04-14/
│  ├─ aula_2026-04-21/
│  └─ aula_2026-05-05/
└─ portefolio/
   ├─ README.md
   └─ mia_rl/
```

Ver [portefolio/README.md](portefolio/README.md) para a descrição detalhada do pacote `mia_rl`.

## Troubleshooting rápido

- Se o kernel não aparecer no VS Code, rode `uv sync` novamente e reabra o editor.
- Se houver erro de versão, confirme com `python --version` e use Python 3.12+.
- Se necessário, recrie o ambiente apagando `.venv/` e executando `uv sync` de novo.
