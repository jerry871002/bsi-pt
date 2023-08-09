# Symbols and Terminologies

There are some symbols and terminologies used in the paper and this project, we briefly describe them here for reference.

## Symbols
- $\pi$ (`pi`): agent policy
- $\tau$ (`tau`): opponent policy
- $\sigma$ (`sigma`): observation or state
- $\phi$ (`phi`): opponent strategy

!!! note
    In the paper, $\phi$ is used to denote opponent strategy indicator, and $\omega$ is used to denote the actual opponent strategy. I.e., when an agent has has a high belief on $\phi_i$, the agent believes the opponent is using $\omega_i$. However, in this project, we use $\phi$ (or `phi` in the code) to denote opponent strategy for simplicity.

## Terminology

- Policy ($\pi$ or $\tau$): a mapping from states to actions
- Strategy ($\phi$): a mapping from a previous observation ($\sigma^{t-1}$) to an opponent’s current policy ($\tau^t$)
