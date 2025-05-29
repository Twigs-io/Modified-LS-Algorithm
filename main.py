# region imports
from AlgorithmImports import *
import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.polynomial import polyfit, polyval
import base64
import io
# endregion

class JumpingRedArmadillo(QCAlgorithm):

    def Initialize(self):
        # Load GRU weights from Dropbox (base64 encoded npz file)
        file_str = self.Download("")
        file_bytes = base64.b64decode(file_str)
        weights = np.load(io.BytesIO(file_bytes), allow_pickle=True)

        # Load GRU layer weights
        self.W_z, self.U_z, self.b_z = weights['W_z'], weights['U_z'], weights['b_z']
        self.W_r, self.U_r, self.b_r = weights['W_r'], weights['U_r'], weights['b_r']
        self.W_h, self.U_h, self.b_h = weights['W_h'], weights['U_h'], weights['b_h']
        self.W_out, self.b_out = weights['W_out'], weights['b_out']

        # Convert to torch tensors
        W_r, W_z, W_h = map(lambda x: torch.tensor(x, dtype=torch.float32), [self.W_r, self.W_z, self.W_h])
        U_r, U_z, U_h = map(lambda x: torch.tensor(x, dtype=torch.float32), [self.U_r, self.U_z, self.U_h])
        b_r, b_z, b_h = map(lambda x: torch.tensor(x, dtype=torch.float32), [self.b_r, self.b_z, self.b_h])

        # Combine weights for GRU format
        weight_ih = torch.cat([W_r, W_z, W_h], dim=0)
        weight_hh = torch.cat([U_r, U_z, U_h], dim=0)
        bias_ih = torch.cat([b_r, b_z, b_h], dim=0)
        bias_hh = torch.zeros_like(bias_ih)

        self.input_size = W_z.shape[1]
        self.hidden_size = W_z.shape[0]

        # Build GRU model and assign weights
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        with torch.no_grad():
            self.gru.weight_ih_l0.copy_(weight_ih)
            self.gru.weight_hh_l0.copy_(weight_hh)
            self.gru.bias_ih_l0.copy_(bias_ih)
            self.gru.bias_hh_l0.copy_(bias_hh)

        # Output linear layer
        self.fc_out = nn.Linear(self.hidden_size, 1)
        with torch.no_grad():
            self.fc_out.weight.copy_(torch.tensor(self.W_out, dtype=torch.float32))
            self.fc_out.bias.copy_(torch.tensor(self.b_out, dtype=torch.float32))

        # Backtest configuration
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 30)
        self.SetCash(100000)

        self.set_warm_up(50)

        # Add SPY options and filter
        option = self.AddOption("SPY", Resolution.HOUR)
        option.SetFilter(-10, 10, timedelta(0), timedelta(180))
        self.symbol = option.Symbol

        # LSM parameters
        self.num_paths = 100
        self.timesteps = 50
        self.volatility = None


    def OnData(self, data):
        # Fetch option chain
        chain = data.OptionChains.get(self.symbol)
        rfr = self.risk_free_interest_rate_model.get_interest_rate(data.time)
        if chain is None:
            return

        # Filter near-the-money, short-dated contracts
        contracts = [
            x for x in chain
            if abs(x.Strike - self.Securities[x.Symbol.Underlying].Price) / self.Securities[x.Symbol.Underlying].Price < 0.03
            and (x.Expiry - self.Time).days <= 30
            and x.AskPrice > 0
        ]
        if not contracts:
            return

        underlying_symbol = contracts[0].Symbol.Underlying
        history = self.History(underlying_symbol, 252, Resolution.HOUR)
        if history.empty:
            return

        # Estimate annualized volatility
        log_returns = np.log(history['close'] / history['close'].shift(1)).dropna()
        self.volatility = np.std(log_returns) * np.sqrt(252)

        S0 = self.Securities[underlying_symbol].Price

        # Choose contract with best value (LSM price / Ask)
        best_contract = max(
            contracts, 
            key=lambda x: (self.LSM(x, S0, x.Strike, rfr) / x.AskPrice) if x.AskPrice > 0 else 0
        )

        # Get LSM-based price
        price = self.LSM(best_contract, S0, best_contract.Strike, rfr)
        price = max(0.01, price) if not np.isnan(price) else 0.01

        self.log(f"LSM: Strike={best_contract.Strike}, Expiry={best_contract.Expiry}, S0={S0}, Sigma={self.volatility}, LSM_Price={price}, Ask={best_contract.AskPrice}")

        # Position sizing
        premium = best_contract.AskPrice * 100
        quantity = int(0.05 * self.Portfolio.Cash / premium)

        if quantity > 0 and not self.Portfolio[best_contract.Symbol].Invested:
            self.MarketOrder(best_contract.Symbol, quantity)


    def GRU_Predict_torch(self, X):
        """
        Run forward pass through GRU + output layer
        X: torch.Tensor of shape (batch, seq_len, input_size)
        Returns: torch.Tensor of shape (batch,)
        """
        self.gru.eval()
        self.fc_out.eval()

        with torch.no_grad():
            out, _ = self.gru(X)
            output = self.fc_out(out[:, -1, :])
            return output.squeeze(-1)


    def LSM(self, contract, S0, K, rfr):
        """
        Longstaff-Schwartz Monte Carlo using GRU to estimate continuation values.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        T = (contract.Expiry - self.Time).days / 365
        dt = T / self.timesteps
        discount_factor = torch.exp(torch.tensor(-rfr * dt, dtype=torch.float32, device=device))
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

        # Simulate paths
        s_j = torch.zeros((self.num_paths, self.timesteps + 1), device=device)
        s_j[:, 0] = S0
        for t in range(1, self.timesteps + 1):
            Z = torch.randn(self.num_paths, device=device)
            s_j[:, t] = s_j[:, t - 1] * torch.exp(
                (rfr - 0.5 * self.volatility**2) * dt + self.volatility * sqrt_dt * Z
            )

        # Initialize payoff at maturity
        if contract.Right == OptionRight.Call:
            v = torch.clamp(s_j[:, -1] - K, min=0)
        else:
            v = torch.clamp(K - s_j[:, -1], min=0)

        # Construct input features: (S_t, K, time_left)
        time_grid = dt * (self.timesteps - torch.arange(self.timesteps + 1, device=device)).unsqueeze(0)
        X_all = torch.zeros((self.num_paths, self.timesteps + 1, self.input_size), device=device)
        X_all[:, :, 0] = s_j
        X_all[:, :, 1] = K
        X_all[:, :, 2] = time_grid

        # Backward induction
        with torch.no_grad():
            for t in range(self.timesteps - 1, 0, -1):
                S_t = s_j[:, t]
                V_next = v * discount_factor

                if contract.Right == OptionRight.Call:
                    itm_mask = S_t > K
                    immediate = torch.clamp(S_t - K, min=0)
                else:
                    itm_mask = S_t < K
                    immediate = torch.clamp(K - S_t, min=0)

                if itm_mask.any():
                    X_seq = X_all[itm_mask, :t + 1, :].to(device)
                    continuation_vals = self.GRU_Predict_torch(X_seq)
                    continuation = torch.zeros_like(S_t)
                    continuation[itm_mask] = continuation_vals
                else:
                    continuation = torch.zeros_like(S_t)

                # Exercise if immediate > continuation
                v = torch.where(immediate > continuation, immediate, V_next)

        return (v.mean() * discount_factor).item()
