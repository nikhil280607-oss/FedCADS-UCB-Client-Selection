# FedCADS-UCB: Client Selection in Federated Learning

**Federated CUSUM-Adaptive Discounted Shapley UCB**

AI211 — Machine Learning | Indian Institute of Technology Ropar

Nikhil Varma (2024AIB1350)  Kush Mistry (2024AIB1368)  Rishi Datt Gupta (2024AIB1377)   · Mentor: Shradha Sharma

---

## What This Project Does

This project studies intelligent client selection for Federated Learning under two challenges that standard algorithms fail to handle together: **extreme non-IID data** (each of 50 clients holds only 2 of 10 digit classes) and **concept drift** (30% of clients are silently poisoned mid-training at round 50).

We implement and compare two baselines — DivFL and S-FedAvg — then propose **FedCADS-UCB**, which combines Shapley-based bandit rewards, a CUSUM drift detector that dynamically adjusts memory speed, client quarantine with exponential backoff, and fairness constraints.

---

## Results

![Task 3: Algorithm Robustness to Concept Drift](results/FinalComparision(FedCADS-USB).png)

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
Selects clients by greedily maximizing gradient diversity via a Facility Location objective. Converges fastest early on, but actively selects poisoned clients during drift because their anomalous gradients appear maximally diverse.

The distance between client updates is:

$$
d(i,j) = \|\Delta_i - \Delta_j\|_2
$$

For a selected set of clients \(S\), the total nearest-representative distance is:

$$
C(S) = \sum_{k=1}^{N} \min_{i \in S} d(k,i)
$$

The greedy step selects the client that most reduces this cost:

$$
i^* = \arg\max_{i \notin S} \left[C(S)-C(S \cup \{i\})\right]
$$

### S-FedAvg
Measures each client's true marginal contribution to validation accuracy using Monte Carlo Shapley values. Maintains a per-client relevance score via exponential moving average (alpha=0.75) and selects probabilistically via softmax. Partially recovers from drift but slowly, because the fixed alpha cannot accelerate forgetting when damage accumulates quickly.

For a coalition \(S\), let its utility be the validation accuracy after applying the coalition's average update:

$$
V(S)=\text{Acc}\left(w_t+\frac{1}{|S|}\sum_{j\in S}\Delta_j\right), \quad S\ne\emptyset; \qquad V(\emptyset)=\text{Acc}(w_t)
$$

The Monte Carlo Shapley estimate averages each client's marginal contribution over \(R\) random permutations:

$$
\widehat{\phi}_i=\frac{1}{R}\sum_{r=1}^{R}\left[V(S_i^{(r)}\cup\{i\})-V(S_i^{(r)})\right]
$$

The relevance score uses an exponential moving average with \(\alpha=0.75\):

$$
R_i^{(t)}=\alpha R_i^{(t-1)}+(1-\alpha)\widehat{\phi}_i^{(t)}
$$

Clients are sampled using softmax probabilities:

$$
P(i)=\frac{e^{R_i}}{\sum_j e^{R_j}}
$$

### FedCADS-UCB (Proposed)

The core novelty is connecting a **CUSUM drift detector** to the **discount factor** of a Shapley-valued bandit so that memory speed adapts automatically in real-time.
**Adaptive discount via CUSUM:**
Accuracy drops are accumulated each round (minus a 0.01 slack to absorb noise). When the accumulator exceeds threshold 0.05, drift is declared and the discount factor gamma smoothly drops from 0.95 (long memory, stable) toward 0.30 (rapid forgetting, fast adaptation). As poisoned clients are isolated and accuracy recovers, gamma automatically returns to 0.95.

$$
g_t=A_{t-1}-A_t
$$

$$
C_t=\max(0, C_{t-1}+g_t-\kappa), \quad S_t=\min\left(\frac{C_t}{h_{\mathrm{alarm}}},1\right)
$$

$$
\gamma_t=\gamma_{\max}-(\gamma_{\max}-\gamma_{\min})S_t
$$

The historical reward and selection count are discounted by \(\gamma_t\), then updated for each selected client:

$$
\widetilde R_k\leftarrow\gamma_t\widetilde R_k, \quad \widetilde N_k\leftarrow\gamma_t\widetilde N_k
$$

$$
\widetilde R_k\leftarrow\widetilde R_k+SV_k, \quad \widetilde N_k\leftarrow\widetilde N_k+1
$$

**Client quarantine with exponential backoff:**
A client whose Shapley value is negative for two consecutive rounds is quarantined. Durations escalate as 10, 20, 40, and 80 rounds for successive offenses. Persistently harmful clients are effectively removed without a permanent ban. A gradual suspicion decay (decrement by 1 per positive round rather than hard reset) prevents adversarial clients from evading quarantine by alternating slightly negative and slightly positive Shapley values.

A selected client is flagged when its Shapley value is below the negative threshold; after two consecutive flagged selections, its quarantine duration is:

$$
SV_k<-\delta_q, \quad \delta_q=0.01
$$

$$
Q_k=T_{\mathrm{quarantine}}\cdot 2^{\min(q_k-1,3)}, \quad T_{\mathrm{quarantine}}=10
$$

For clients already in quarantine, the remaining duration decreases by one each round:

$$
Q_k\leftarrow Q_k-1
$$

**Fairness via virtual queue:**
Each client accumulates a fairness debt when not selected, targeting a minimum selection rate of 1% per round. Final score = 90% UCB term + 10% fairness debt, ensuring all digit classes remain covered throughout training.

Average historical reward and exploration bonus:

$$
\widehat{\mu}_k=\frac{\widetilde R_k}{\widetilde N_k}, \quad B_k=\sqrt{\frac{\ln(t+1)}{\widetilde N_k}}
$$

$$
UCB_k=\widehat{\mu}_k+B_k
$$

Fairness debt and the final selection score:

$$
D_k=\max(0,D_k+c_k-b_k), \quad b_k=\begin{cases}1 & \text{if client }k\text{ is selected}\\0 & \text{otherwise}\end{cases}
$$

$$
Score_k=(1-\beta)UCB_k+\beta D_k=0.9UCB_k+0.1D_k, \quad \beta=0.10
$$

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
## References
1. Balakrishnan et al., "Diverse Client Selection for Federated Learning via Submodular Maximization," ICLR 2022
2. Nagalapatti & Narayanam, "Game of Gradients: Mitigating Irrelevant Clients in Federated Learning," AAAI 2021
3. Fouad et al., "Combinatorial Semi-Bandit in the Non-Stationary Environment," UAI 2020
4. Jothimurugesan et al., "Federated Learning under Distributed Concept Drift," AISTATS 2023
5. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017

---
## License

MIT License — see LICENSE for details.
