# Portefólio — Aprendizagem por Reforço

**Universidade do Minho — Mestrado em Inteligência Artificial**

Aluno: Diego Jefferson Mendes Silva — pg59999

---

## Configuração do Ambiente

O projeto utiliza [uv](https://github.com/astral-sh/uv) para gestão do ambiente virtual e das dependências Python (≥ 3.12).

```bash
# Instalar uv (caso não esteja instalado)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Criar o ambiente virtual e instalar as dependências
uv sync

# Ativar o ambiente virtual
source .venv/bin/activate
```

As dependências do projeto estão definidas em `pyproject.toml`:

| Pacote | Utilização |
|---|---|
| `numpy` | Cálculo numérico (vetores de pesos, features, retornos) |
| `torch` | Agente SARSA com aproximação de função via PyTorch |
| `matplotlib` | Geração dos gráficos de treino e políticas |
| `ipykernel` | Suporte aos notebooks Jupyter |

---

## Estrutura do Pacote `mia_rl`

```
mia_rl/
├── core/          # Classes base abstratas
├── mdps/          # Interface para MDPs tabulares (modelo conhecido)
├── envs/          # Ambientes de RL
├── agents/
│   ├── control/   # Agentes de controlo (aprendizagem da política)
│   ├── prediction/# Agentes de predição (estimação de V)
│   └── planning/  # Planeamento (MCTS)
├── features/      # Extração de features para aproximação de função
├── policies/      # Políticas fixas e interativas
├── experiments/   # Funções de treino e avaliação
├── scripts/       # Pontos de entrada CLI
└── notebooks/     # Notebooks demonstrativos
```

### `core/`

| Ficheiro | Descrição |
|---|---|
| `base.py` | Define as abstrações fundamentais do framework: `Transition` (tuplo `s, a, r, s', done`), `Episode` (sequência de transições), `Environment`, `Policy`, `Agent`, e `PredictionAgent`. |

### `mdps/`

| Ficheiro | Descrição |
|---|---|
| `base.py` | Define `TabularMDP`, interface abstrata para MDPs com modelo conhecido, usada em programação dinâmica. Requer implementar `states()`, `possible_actions()`, `is_terminal()` e `transitions()`. |

### `envs/`

| Ficheiro | Descrição |
|---|---|
| `blackjack.py` | Ambiente de Blackjack (estilo casino). Estado: `(soma_jogador, carta_dealer, ás_usável)`. Ações: `hit` / `stick`. Recompensa: +1 (vitória), −1 (derrota), 0 (empate). |
| `windy_gridworld.py` | Windy Gridworld 7×10 (Sutton & Barto, Exemplo 6.5). Vento empurra o agente para cima em certas colunas. Recompensa: −1 por passo; episódio termina ao atingir o objetivo `(3,7)`. |
| `tictactoe.py` | Jogo do Galo para dois jogadores. Tabuleiro como 9-tuplo. Inclui deteção de vitória/empate e suporte a dois jogadores alternados. |

### `agents/control/`

| Ficheiro | Descrição |
|---|---|
| `base.py` | `ControlAgent` abstrato com interface `select_action()`, `update_transition()`, `action_value_of()` e `end_episode()`. |
| `sarsa.py` | **SARSA tabular** (controlo on-policy TD). Q-table com `defaultdict`; atualização: $Q(s,a) \mathrel{+}= \alpha(r + \gamma Q(s',a') - Q(s,a))$. |
| `monte_carlo.py` | **Monte Carlo de primeira visita** (on-policy, ε-greedy). Retornos calculados no fim do episódio com atualização incremental dos Q-valores. |
| `n_step_sarsa.py` | **SARSA n-passos**. Buffer deslizante de transições; retorno n-passos com bootstrap no último Q-valor guardado. |
| `linear_sarsa.py` | **SARSA semi-gradiente com aproximação linear (NumPy)**. $\hat{q}(s,a) = \mathbf{w} \cdot \phi(s,a)$; atualização: $\mathbf{w} \mathrel{+}= \alpha \delta \phi(s,a)$. |
| `torch_sarsa.py` | **SARSA semi-gradiente com PyTorch**. Demonstra dois estilos de atualização: manual (`loss.backward()` + gradiente à mão) e via `torch.optim.SGD`. Target sempre detachado (semi-gradiente). |
| `reinforce.py` | **REINFORCE (gradiente de política Monte Carlo)** para Jogo do Galo. Política softmax com codificação perspetiva-relativa; suporte a regularização por entropia. |

### `agents/prediction/`

| Ficheiro | Descrição |
|---|---|
| `monte_carlo.py` | **Monte Carlo de primeira visita** para Blackjack. Atualização de $V(s)$ com média incremental. |
| `td.py` | **TD(0)** para Blackjack. Atualização online por passo: $V(s) \mathrel{+}= \alpha(r + \gamma V(s') - V(s))$. |
| `linear_td.py` | **TD(0) semi-gradiente com aproximação linear** para Windy Gridworld. $\hat{v}(s) = \mathbf{w} \cdot \phi(s)$; devolve o erro TD por passo. |

### `agents/planning/`

| Ficheiro | Descrição |
|---|---|
| `mcts.py` | **Monte Carlo Tree Search (MCTS)** para Jogo do Galo. Implementa as 4 fases clássicas (seleção UCB1, expansão, simulação aleatória, retropropagação). Constrói uma nova árvore em cada chamada. |

### `features/`

| Ficheiro | Descrição |
|---|---|
| `tictactoe.py` | Feature 27-dimensional perspetiva-relativa para Jogo do Galo. Cada célula codificada como `[peça_própria, peça_adversário, vazia]`, permitindo reutilizar os mesmos pesos para X e O. |
| `windy_gridworld.py` | **Tile coding** para Windy Gridworld: 4 sobreposições com tiles 2×2, produzindo um vetor esparso de 96 dimensões. `state_action_features()` gera encoding de 384 dimensões (96 por ação). |

### `policies/`

| Ficheiro | Descrição |
|---|---|
| `blackjack.py` | `ThresholdPolicy`: política fixa para Blackjack — pede carta se `soma < limiar` (padrão: 20). |
| `tictactoe.py` | `random_action` (movimento aleatório legal) e `human_policy` (leitura de stdin, índice 1–9). |

### `experiments/`

| Ficheiro | Descrição |
|---|---|
| `training.py` | Harness de treino para Blackjack. `generate_episode()` e `train_prediction_agent()` com snapshots da função de valor a intervalos configuráveis. |
| `control.py` | Harness para agentes de controlo no Windy Gridworld. `run_control_episode()` (loop online SARSA), `greedy_policy_from_agent()` e `greedy_path()`. |
| `fa_training.py` | Harness para agentes com aproximação de função. Recolhe erros TD por episódio; inclui `run_linear_td_episode()` para TD(0) com política de comportamento fixa. |
| `tictactoe.py` | Utilitários de jogo: `play_game()` entre dois agentes/políticas e `play_game_vs_human()` com input via stdin. |
| `mcts_tictactoe.py` | Avaliação de MCTS: `evaluate_vs_random()` (taxa de vitória/empate/derrota) e `evaluate_mcts_vs_reinforce()`. |
| `reinforce_tictactoe.py` | Treino e avaliação de REINFORCE para Jogo do Galo. Suporta auto-jogo e treino contra oponente aleatório; retorna métricas de desempenho. |

### `scripts/`

Pontos de entrada CLI executáveis diretamente (ex.: `python -m mia_rl.scripts.run_windy_gridworld_sarsa`).

| Ficheiro | Descrição |
|---|---|
| `run_blackjack_prediction.py` | Treina Monte Carlo e TD(0) para Blackjack; guarda heatmaps da função de valor. |
| `run_windy_gridworld_sarsa.py` | SARSA tabular no Windy Gridworld; guarda curvas de treino e grelha de política. |
| `run_windy_gridworld_mc_control.py` | Monte Carlo control no Windy Gridworld. |
| `run_windy_gridworld_n_step_sarsa.py` | SARSA n-passos (argumento `--n-steps` configurável). |
| `run_windy_gridworld_linear_sarsa.py` | SARSA linear (NumPy) com aproximação de função; guarda curva de erro TD, heatmap de valor e grelha de política. |
| `run_windy_gridworld_linear_td.py` | TD(0) linear: pré-treina SARSA para obter política de comportamento, depois avalia com `LinearTD0`. |
| `run_windy_gridworld_torch_sarsa.py` | Compara os quatro agentes (Tabular, Linear NumPy, Torch manual, Torch SGD) numa curva de comprimento de episódio suavizada. |

### `notebooks/`

| Ficheiro | Descrição |
|---|---|
| `TicTacToe_Demo.ipynb` | Demonstração interativa do Jogo do Galo com políticas aleatória e humana. |
| `TicTacToe_MCTS.ipynb` | Avaliação e visualização do agente MCTS. |
| `TicTacToe_PolicyGradient.ipynb` | Treino e análise do agente REINFORCE para Jogo do Galo. |

