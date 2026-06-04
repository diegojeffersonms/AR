Três papéis diferentes que um agente pode desempenhar:

prediction
Aprende a estimar valor (ex.: (V(s)), (Q(s,a))) sem necessariamente melhorar a política.
Objetivo: avaliar “quão bom” é estar num estado ou fazer uma ação.

control
Aprende uma política ótima (ou melhor) para maximizar retorno.
Objetivo: decidir ações. Normalmente usa estimativas de valor para melhorar comportamento (ex.: SARSA, Monte Carlo control, REINFORCE).

planning
Escolhe ações usando um modelo do ambiente (ou simulação), em vez de aprender só por experiência direta.
Objetivo: “pensar antes de agir” (ex.: MCTS).