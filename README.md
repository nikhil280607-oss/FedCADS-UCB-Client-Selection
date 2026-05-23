# FedCADS-UCB: Client Selection in Federated Learning

**Federated CUSUM-Adaptive Discounted Shapley UCB**

AI211 — Machine Learning Theory | Indian Institute of Technology Ropar

Kush Mistry (2024AIB1368) · Rishi Datt Gupta (2024AIB1377) · I. Nikhil Varma (2024AIB1350) · Mentor: Shradha Sharma

---

## What This Project Does

This project studies intelligent client selection for Federated Learning under two challenges that standard algorithms fail to handle together: **extreme non-IID data** (each of 50 clients holds only 2 of 10 digit classes) and **concept drift** (30% of clients are silently poisoned mid-training at round 50).

We implement and compare two baselines — DivFL and S-FedAvg — then propose **FedCADS-UCB**, which combines Shapley-based bandit rewards, a CUSUM drift detector that dynamically adjusts memory speed, client quarantine with exponential backoff, and fairness constraints.

---

## Results

![Task 3: Algorithm Robustness to Concept Drift](results/task3_fedcads_comparison.png)

| Algorithm | Pre-Drift (R50) | Round 60 | Final (R100) |
|---|---|---|---|
| DivFL | 95.87% | 57.46% | 58.43% |
| S-FedAvg | 95.20% | 78.31% | ~80% |
| **FedCADS-UCB (ours)** | **94.60%** | **~90%** | **96.95%** |

Drift injected at round 50 via label shift: y_new = (y + 1) mod 10 on clients 0-14.

---

## Repository Structure

```
FedCADS-UCB-Client-Selection/
├── src/
│   ├── DivFL.py            # Client update logic and DivFL selection
│   ├── data_setup.py       # MNIST loading and non-IID partitioning
│   ├── models.py           # LR, SVM, DNN, Simple CNN architectures
│   ├── shapley.py          # Monte Carlo Shapley estimation, S-FedAvg
│   ├── fedcads.py          # FedCADS-UCB: CUSUM, UCB, quarantine, fairness
│   ├── train_task1.py      # Comparison of DivFL and S-FedAvg algoriths using four different models   
│   ├── train_task2.py      # Comparison of DivFL and S-FedAvg after Concept Drift
│   └── train_task3.py      # Full three-algorithm comparison training loop
├── results/                # Output plots for all three tasks
├── docs/                   # IEEE project report (PDF)
└── README.md
```

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | MNIST (60K train / 10K test) |
| Clients | 50 total, 10 selected per round |
| Data split | Non-IID: 2 shards per client, at most 2 digit classes |
| Rounds | 100 communication rounds |
| Local training | 3 epochs, SGD, lr=0.01, batch=16 |
| Shapley estimation | R=5 Monte Carlo permutations |
| Models tested | Logistic Regression, SVM, DNN, Simple CNN |

---

## The Three Algorithms

### DivFL
Selects clients by greedily maximizing gradient diversity via a Facility Location objective. Guarantees (1 - 1/e)-optimal coverage of the client population. Converges fastest early on, but actively selects poisoned clients during drift because their anomalous gradients appear maximally diverse.

### S-FedAvg
Measures each client's true marginal contribution to validation accuracy using Monte Carlo Shapley values. Maintains a per-client relevance score via exponential moving average (alpha=0.75) and selects probabilistically via softmax. Partially recovers from drift but slowly, because the fixed alpha cannot accelerate forgetting when damage accumulates quickly.

### FedCADS-UCB (Proposed)

The core novelty is connecting a **CUSUM drift detector** to the **discount factor** of a Shapley-valued bandit so that memory speed adapts automatically in real-time.

**Adaptive discount via CUSUM:**
Accuracy drops are accumulated each round (minus a 0.01 slack to absorb noise). When the accumulator exceeds threshold 0.05, drift is declared and the discount factor gamma smoothly drops from 0.95 (long memory, stable) toward 0.30 (rapid forgetting, fast adaptation). As poisoned clients are isolated and accuracy recovers, gamma automatically returns to 0.95.

**Client quarantine with exponential backoff:**
A client whose Shapley value is negative for two consecutive rounds is quarantined. Durations escalate as 10, 20, 40, and 80 rounds for successive offenses. Persistently harmful clients are effectively removed without a permanent ban. A gradual suspicion decay (decrement by 1 per positive round rather than hard reset) prevents adversarial clients from evading quarantine by alternating slightly negative and slightly positive Shapley values.

**Fairness via virtual queue:**
Each client accumulates a fairness debt when not selected, targeting a minimum selection rate of 1% per round. Final score = 90% UCB term + 10% fairness debt, ensuring all digit classes remain covered throughout training.

---

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

**Run the full Task 3 comparison:**

```bash
cd src
python train_task3.py
```

Downloads MNIST automatically, trains all three algorithms for 100 rounds each with drift at round 50, and saves the comparison plot. Expected runtime: approximately 80-90 minutes on GPU.

---

## Theoretical Guarantees

FedCADS-UCB achieves near-optimal convergence in both regimes simultaneously:

- **Stable phase** (gamma near 0.95): regret O(sqrt T), matching standard Shapley-UCB
- **Drift phase** (gamma near 0.30): regret O(T^(2/3)), matching CUCB-SW

No single existing algorithm achieves both. S-FedAvg achieves the stable bound but is suboptimal under drift. CUCB-SW achieves the drift bound but wastes information in stable periods. FedCADS-UCB interpolates continuously between the two regimes based on live CUSUM output.

---

## References

1. Balakrishnan et al., "Diverse Client Selection for Federated Learning via Submodular Maximization," ICLR 2022
2. Nagalapatti & Narayanam, "Game of Gradients: Mitigating Irrelevant Clients in Federated Learning," AAAI 2021
3. Fouad et al., "Combinatorial Semi-Bandit in the Non-Stationary Environment," UAI 2020
4. Jothimurugesan et al., "Federated Learning under Distributed Concept Drift," AISTATS 2023
5. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017

---

## License

MIT License — see LICENSE for details.
