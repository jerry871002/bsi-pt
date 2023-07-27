# Bayesian Strategy Inference plus Policy Tracking (BSI-PT)

Welcome to the BSI-PT project! This project implements the experiments presented in the paper [Opponent Exploitation Based on Bayesian Strategy Inference and Policy Tracking](https://ieeexplore.ieee.org/document/10148618).

In the experiments, we compare our BSI-PT framework with previous BPR[^bpr-def] variants including [BPR+](https://researchspace.csir.co.za/dspace/bitstream/handle/10204/9091/Hernandez-Leal_2016.pdf?sequence=1&isAllowed=y)[^bpr], [DPN-BPR+](https://drive.google.com/file/d/1FMxWLF3hAgKTomp-foczAY3ppfF-GE-2/view)[^dpn-bpr], and [Bayes-OKR](https://www.sciencedirect.com/science/article/abs/pii/S0950705122001605)[^okr] in the [extended batter vs. pitcher game (EBvPG)](environments/baseball_game.md) defined in the paper.

## Getting Started

To reproduce the results in the paper, run the following command:

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

The default of `[NUM_RUNS]` is quite large since we want to make sure the results are statistically significant, so it's recommended to set it to a smaller number (e.g. 10) for a quick run.

The figures will be generated and store in `src/fig/`, while the source data will be stored in `src/data/` and `src/csv/`.

If you want to run the experiments individually or inspect each algorithm in detail, see [this document](detail.md).

## Extras

In addition to the EBvPG environment, which is under the `baseball_game/` directory in this repository, we also implemented three other environments inspired by the Bayes-OKR[^okr] paper, including [Grid World](environments/grid_world.md), [Navigation Game](environments/navigation_game.md), and [Soccer Game](environments/soccer_game.md). In these three environments, [Bayes-ToMoP](https://arxiv.org/pdf/1809.04240)[^tom] and BSI (BSI-PT without policy tracking capability) agents are also presented.

[^bpr-def]: Bayesian Policy Reuse
[^bpr]: Hernandez-Leal, Pablo, et al. "Identifying and tracking switching, non-stationary opponents: A Bayesian approach." (2016).
[^dpn-bpr]: Zheng, Yan, et al. "Efficient policy detecting and reusing for non-stationarity in Markov games." (2021).
[^tom]: Yang, Tianpei, et al. "Towards efficient detection and optimal response against sophisticated opponents." (2018).
[^okr]: Chen, Hao, et al. "Accurate policy detection and efficient knowledge reuse against multi-strategic opponents." (2022).
