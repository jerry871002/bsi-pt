# Opponent Exploitation Based on Bayesian Strategy Inference and Policy Tracking

This repository implements the experiments presented in the paper "[Opponent Exploitation Based on Bayesian Strategy Inference and Policy Tracking](https://ieeexplore.ieee.org/document/10148618)."

We have implemented six BPR-based algorithms, including
- Our Works
    - BSI (Bayesian Strategy Inference)
    - BSI-PT (Bayesian Strategy Inference with Policy Tracking)
- Previous Works
    - [BPR+](https://researchspace.csir.co.za/dspace/bitstream/handle/10204/9091/Hernandez-Leal_2016.pdf?sequence=1&isAllowed=y)[^bpr]
    - [DPN-BPR+](https://drive.google.com/file/d/1FMxWLF3hAgKTomp-foczAY3ppfF-GE-2/view)[^dpn-bpr]
    - [Bayes-ToMoP](https://arxiv.org/pdf/1809.04240)[^tom]
    - [Bayes-OKR](https://www.sciencedirect.com/science/article/abs/pii/S0950705122001605)[^okr]

Benchmarked in three environments
- [Grid World](src/grid_world/)
- [Navigation Game](src/navigation_game/)
- [Soccer Game](src/soccer_game/)

## Getting Started

To reproduce the results in the paper, run the following command:

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

the figures will be generated and store in `src/fig/`, while the source data will be stored in `src/data/` and `src/csv/`.

If you want to run the experiments individually or inspect each algorithm in detail, refer to [this document](src/README.md)

[^bpr]: Hernandez-Leal, Pablo, et al. "Identifying and tracking switching, non-stationary opponents: A Bayesian approach." (2016).
[^dpn-bpr]: Zheng, Yan, et al. "Efficient policy detecting and reusing for non-stationarity in Markov games." (2021).
[^tom]: Yang, Tianpei, et al. "Towards efficient detection and optimal response against sophisticated opponents." (2018).
[^okr]: Chen, Hao, et al. "Accurate policy detection and efficient knowledge reuse against multi-strategic opponents." (2022).
