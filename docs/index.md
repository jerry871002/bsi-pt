# Bayesian Strategy Inference plus Policy Tracking (BSI-PT)

This project implements the experiments presented in the paper [Opponent Exploitation Based on Bayesian Strategy Inference and Policy Tracking](https://ieeexplore.ieee.org/document/10148618).

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

To reproduce the results in the paper, use the `scripts/run_exps_and_plot.sh` script. There are two ways to run the script.

### Run locally

Make sure you have Python 3.8+ installed, then install the dependencies in `requirements.txt` using `pip`.

```
pip install -r requirements.txt
```

Then run the following command:

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

### Run in Docker

Build the image before the first run or after you make a change to the code.

```
docker build -t bsi-pt .
```

Run a container using the image and mount the `src/` directory to see the results.

```
docker run -it -v $(pwd)/src:/app/src bsi-pt
```

You will enter an interactive shell inside the container, run the experiments using the following command:

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

The raw data will be stored in `src/data/` in `.pkl` format. The figures will be generated from those data and stored in `src/fig/`, then the data points in the figures will be stored in CSV format in `src/csv/`.

If you want to run the experiments individually or inspect each algorithm in detail, see [this document](src.md).

[^bpr]: Hernandez-Leal, Pablo, et al. "Identifying and tracking switching, non-stationary opponents: A Bayesian approach." (2016).
[^dpn-bpr]: Zheng, Yan, et al. "Efficient policy detecting and reusing for non-stationarity in Markov games." (2021).
[^tom]: Yang, Tianpei, et al. "Towards efficient detection and optimal response against sophisticated opponents." (2018).
[^okr]: Chen, Hao, et al. "Accurate policy detection and efficient knowledge reuse against multi-strategic opponents." (2022).
