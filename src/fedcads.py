import numpy as np

class FedCADS:
    def __init__(self, num_clients, fairness_rates, beta=0.1):
        self.num_clients = num_clients
        self.c_k = np.array(fairness_rates)
        self.beta = beta

        self.gamma_max = 0.95
        self.gamma_min = 0.30
        self.kappa = 0.01
        self.h_alarm = 0.05
        self.delta_q = 0.01
        self.T_quarantine = 10

        self.R_tilde = np.zeros(num_clients)
        self.N_tilde = np.full(num_clients, 1e-5)
        self.D_k = np.zeros(num_clients)

        self.Q_k = np.zeros(num_clients, dtype=int)
        self.consecutive_negative_sv = np.zeros(num_clients, dtype=int)

        # Step 1: Track quarantine counts for exponential backoff
        self.quarantine_count = np.zeros(num_clients, dtype=int)

        self.C_t = 0.0
        self.gamma = self.gamma_max
        self.round_t = 0

    def update_drift_and_gamma(self, accuracy_drop):
        # Step 2: Update drift signal and adjust gamma
        self.round_t += 1
        g_t = accuracy_drop
        self.C_t = max(0.0, self.C_t + g_t - self.kappa)
        drift_signal = min(self.C_t / self.h_alarm, 1.0)
        self.gamma = self.gamma_max - ((self.gamma_max - self.gamma_min) * drift_signal)
        return self.gamma, drift_signal

    def get_selection_scores(self):
        # Step 3: Calculate UCB selection scores for active clients
        scores = np.full(self.num_clients, -np.inf)
        for k in range(self.num_clients):
            if self.Q_k[k] == 0:
                mu_hat = self.R_tilde[k] / self.N_tilde[k]
                exploration_bonus = np.sqrt(np.log(self.round_t + 1) / self.N_tilde[k])
                ucb_term = mu_hat + exploration_bonus
                scores[k] = (1 - self.beta) * ucb_term + self.beta * self.D_k[k]
        return scores

    def update_client_stats(self, selected_clients, shapley_values):
        # Step 4: Update fairness deficit and decay historical stats
        for k in range(self.num_clients):
            if self.Q_k[k] > 0:
                self.Q_k[k] -= 1
            else:
                b_k = 1 if k in selected_clients else 0
                self.D_k[k] = max(0.0, self.D_k[k] + self.c_k[k] - b_k)

            self.R_tilde[k] *= self.gamma
            self.N_tilde[k] *= self.gamma

        # Step 5: Process Shapley values and handle client quarantine
        for k, sv in zip(selected_clients, shapley_values):
            self.R_tilde[k] += sv
            self.N_tilde[k] += 1.0

            if sv < -self.delta_q:
                self.consecutive_negative_sv[k] += 1
                if self.consecutive_negative_sv[k] >= 2:
                    # Apply exponential backoff for quarantine duration (cap at 4x)
                    self.quarantine_count[k] += 1
                    duration = self.T_quarantine * (2 ** min(self.quarantine_count[k] - 1, 3))
                    self.Q_k[k] = duration
                    self.consecutive_negative_sv[k] = 0
                    print(f"  [!] FedCADS ALERT: Client {k} quarantined for {duration} rounds "
                          f"(offense #{self.quarantine_count[k]}).", flush=True)
            else:
                # Gradual decay for consecutive negative SVs to prevent gaming the system
                self.consecutive_negative_sv[k] = max(0, self.consecutive_negative_sv[k] - 1)

    def get_quarantined_clients(self):
        # Step 6: Return list of currently quarantined clients
        return [k for k in range(self.num_clients) if self.Q_k[k] > 0]
