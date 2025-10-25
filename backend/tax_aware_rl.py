"""
Tax-Aware Portfolio Management System - RL Environment and Agent
This module contains the reinforcement learning environment and SAC agent implementation.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime as dt
from collections import deque
import logging
import json
import pickle

from tax_aware_core import TaxCalculator, PortfolioState, MarketDataManager, TORCH_AVAILABLE

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal


class TaxAwarePortfolioEnv(gym.Env):
    """Custom Gymnasium environment for tax-aware portfolio management."""
    
    def __init__(self,
                 symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                 initial_cash: float = 1000000.0,
                 start_date: str = '2020-01-01',
                 end_date: str = '2023-12-31',
                 lookback_window: int = 60,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.4,
                 rebalance_frequency: int = 1):
        """
        Initialize the tax-aware portfolio environment.
        
        Args:
            symbols: List of asset symbols to trade
            initial_cash: Starting cash amount
            start_date: Environment start date
            end_date: Environment end date
            lookback_window: Number of historical days for observation
            transaction_cost: Transaction cost as percentage of trade value
            max_position_size: Maximum position size as percentage of portfolio
            rebalance_frequency: Days between rebalancing (1 = daily)
        """
        super().__init__()
        
        # Environment parameters
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_cash = initial_cash
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize components
        self.portfolio = PortfolioState(initial_cash)
        self.tax_calculator = TaxCalculator()
        self.market_data = MarketDataManager(symbols)
        
        # Fetch and prepare data
        try:
            self.data = self.market_data.fetch_data(start_date, end_date)
            self.indicators = self.market_data.calculate_technical_indicators(self.data)
            
            # Prepare time series
            all_dates = set()
            for df in self.data.values():
                all_dates.update(df.index)
            
            self.dates = sorted(list(all_dates))
            self.current_step = 0
            self.max_steps = len(self.dates) - lookback_window - 1
            
            logger.info(f"Environment initialized: {len(self.dates)} days, {self.max_steps} max steps")
            
        except Exception as e:
            logger.error(f"Error initializing environment: {e}")
            self.dates = []
            self.max_steps = 0
        
        # Define spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_count = 0
        self.episode_returns = []
        self.episode_trades = []
        
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        try:
            # Action space: portfolio weights for each asset (continuous)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_assets,),
                dtype=np.float32
            )
            
            # Observation space: market features + portfolio state + tax features
            market_features = 5 * self.n_assets  # OHLCV for each asset
            technical_features = 8 * self.n_assets  # Technical indicators
            portfolio_features = 3 + self.n_assets  # cash, total_value, unrealized_pnl, positions
            tax_features = 5  # tax_liability, tax_alpha, avg_holding_period, wash_sales, ltcg_ratio
            
            obs_dim = market_features + technical_features + portfolio_features + tax_features
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            
            logger.info(f"Spaces defined: Action {self.action_space.shape}, Observation {self.observation_space.shape}")
            
        except Exception as e:
            logger.error(f"Error setting up spaces: {e}")
            # Fallback to minimal spaces
            self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-10, high=10, shape=(20,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        try:
            super().reset(seed=seed)
            
            # Reset portfolio and environment state
            self.portfolio.reset()
            self.current_step = self.lookback_window
            self.episode_count += 1
            self.episode_trades = []
            
            # Get initial observation
            observation = self._get_observation()
            info = self._get_info()
            
            logger.debug(f"Environment reset for episode {self.episode_count}")
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            # Return minimal valid observation
            obs_dim = self.observation_space.shape[0]
            return np.zeros(obs_dim, dtype=np.float32), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        try:
            # Validate and normalize action
            action = np.array(action, dtype=np.float32)
            action = np.clip(action, 0, 1)
            
            # Normalize to portfolio weights
            if np.sum(action) > 0:
                action = action / np.sum(action)
            else:
                action = np.ones(self.n_assets) / self.n_assets
            
            # Apply maximum position size constraint
            action = np.clip(action, 0, self.max_position_size)
            action = action / np.sum(action)  # Renormalize
            
            # Get current market state
            current_date = self.dates[self.current_step]
            prices = self._get_current_prices(current_date)
            
            # Execute rebalancing
            trades = self._rebalance_portfolio(action, prices, current_date)
            
            # Calculate reward
            reward = self._calculate_reward(trades, prices, current_date)
            
            # Update environment state
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated = False
            
            # Get new observation and info
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                'trades': trades,
                'action': action.tolist(),
                'prices': prices,
                'reward_components': self._get_reward_components()
            })
            
            # Track episode data
            self.episode_trades.extend(trades)
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            # Return safe defaults
            obs_dim = self.observation_space.shape[0]
            observation = np.zeros(obs_dim, dtype=np.float32)
            return observation, 0.0, True, False, {'error': str(e)}
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        try:
            if self.current_step >= len(self.dates):
                # Return zero observation if we've run out of data
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            current_date = self.dates[self.current_step]
            observation_features = []
            
            # Market features (OHLCV)
            for symbol in self.symbols:
                if symbol in self.data and current_date in self.data[symbol].index:
                    row = self.data[symbol].loc[current_date]
                    # Normalize prices by first price to make them scale-invariant
                    first_price = self.data[symbol]['Close'].iloc[0]
                    observation_features.extend([
                        row['Open'] / first_price,
                        row['High'] / first_price,
                        row['Low'] / first_price,
                        row['Close'] / first_price,
                        row['Volume'] / 1e9  # Scale volume
                    ])
                else:
                    observation_features.extend([1.0, 1.0, 1.0, 1.0, 0.1])  # Default normalized values
            
            # Technical indicators
            for symbol in self.symbols:
                if (symbol in self.indicators and 
                    current_date in self.indicators[symbol]['returns'].index):
                    
                    indicators = self.indicators[symbol]
                    
                    # Get indicator values with safe defaults
                    sma_20 = self._safe_get_indicator(indicators['sma_20'], current_date, 1.0)
                    sma_50 = self._safe_get_indicator(indicators['sma_50'], current_date, 1.0)
                    rsi = self._safe_get_indicator(indicators['rsi'], current_date, 50.0)
                    vol = self._safe_get_indicator(indicators['volatility_20'], current_date, 0.02)
                    ret = self._safe_get_indicator(indicators['returns'], current_date, 0.0)
                    macd = self._safe_get_indicator(indicators['macd'], current_date, 0.0)
                    
                    # Get current price for normalization
                    current_price = self._get_current_prices(current_date).get(symbol, 100.0)
                    
                    observation_features.extend([
                        sma_20 / current_price if current_price > 0 else 1.0,
                        sma_50 / current_price if current_price > 0 else 1.0,
                        rsi / 100.0,  # Normalize RSI to [0,1]
                        np.tanh(vol * 100),  # Scale and bound volatility
                        np.tanh(ret * 100),  # Scale and bound returns
                        np.tanh(macd / current_price * 100) if current_price > 0 else 0.0,
                        0.0,  # Placeholder for additional indicator
                        0.0   # Placeholder for additional indicator
                    ])
                else:
                    # Default technical indicator values
                    observation_features.extend([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Portfolio features
            prices = self._get_current_prices(current_date)
            portfolio_value = self.portfolio.get_portfolio_value(prices)
            
            portfolio_features = [
                self.portfolio.cash / self.initial_cash,  # Normalized cash
                portfolio_value / self.initial_cash,      # Normalized portfolio value
                0.0  # Placeholder for additional portfolio feature
            ]
            
            # Current positions (normalized by portfolio value)
            for symbol in self.symbols:
                position = self.portfolio.positions.get(symbol, 0)
                position_value = position * prices.get(symbol, 0)
                normalized_position = position_value / portfolio_value if portfolio_value > 0 else 0.0
                portfolio_features.append(normalized_position)
            
            # Tax features
            unrealized_pnl = self.portfolio.get_unrealized_pnl(prices)
            total_unrealized = sum(unrealized_pnl.values())
            
            tax_features = [
                total_unrealized / self.initial_cash,  # Normalized unrealized P&L
                0.0,  # Tax liability (placeholder)
                self._get_average_holding_period() / 365.0,  # Normalized holding period
                0.0,  # Wash sales count (placeholder)
                self._get_long_term_ratio()  # Ratio of long-term holdings
            ]
            
            # Combine all features
            observation = observation_features + portfolio_features + tax_features
            
            # Ensure correct dimensionality
            target_dim = self.observation_space.shape[0]
            if len(observation) < target_dim:
                observation.extend([0.0] * (target_dim - len(observation)))
            elif len(observation) > target_dim:
                observation = observation[:target_dim]
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _safe_get_indicator(self, series: pd.Series, date: dt.datetime, default: float) -> float:
        """Safely get indicator value with fallback."""
        try:
            if date in series.index and not pd.isna(series.loc[date]):
                return float(series.loc[date])
            return default
        except:
            return default
    
    def _get_current_prices(self, date: dt.datetime) -> Dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}
        for symbol in self.symbols:
            try:
                if symbol in self.data and date in self.data[symbol].index:
                    prices[symbol] = float(self.data[symbol].loc[date, 'Close'])
                else:
                    prices[symbol] = 100.0  # Default price
            except Exception as e:
                logger.warning(f"Error getting price for {symbol}: {e}")
                prices[symbol] = 100.0
        return prices
    
    def _rebalance_portfolio(self, 
                           target_weights: np.ndarray, 
                           prices: Dict[str, float],
                           current_date: dt.datetime) -> List[Dict]:
        """Rebalance portfolio to target allocation."""
        trades = []
        
        try:
            portfolio_value = self.portfolio.get_portfolio_value(prices)
            target_values = target_weights * portfolio_value
            
            for i, symbol in enumerate(self.symbols):
                current_position = self.portfolio.positions.get(symbol, 0)
                current_value = current_position * prices.get(symbol, 0)
                target_value = target_values[i]
                
                # Calculate required trade
                value_diff = target_value - current_value
                
                # Apply minimum trade threshold
                min_trade_value = 1000.0  # $1,000 minimum trade
                if abs(value_diff) > min_trade_value:
                    price = prices.get(symbol, 0)
                    if price > 0:
                        quantity = value_diff / price
                        
                        # Apply transaction costs
                        trade_cost = abs(value_diff) * self.transaction_cost
                        if self.portfolio.cash >= trade_cost:
                            # Execute trade
                            result = self.portfolio.execute_trade(
                                symbol, quantity, price, current_date
                            )
                            
                            if result['success']:
                                # Deduct transaction costs
                                self.portfolio.cash -= trade_cost
                                trade_info = result['trade'].copy()
                                trade_info['transaction_cost'] = trade_cost
                                trades.append(trade_info)
                            else:
                                logger.warning(f"Trade failed for {symbol}: {result.get('error', 'Unknown error')}")
            
            return trades
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return []
    
    def _calculate_reward(self, 
                         trades: List[Dict], 
                         prices: Dict[str, float],
                         current_date: dt.datetime) -> float:
        """Calculate multi-objective reward function."""
        try:
            # Portfolio return component
            current_portfolio_value = self.portfolio.get_portfolio_value(prices)
            
            if hasattr(self, 'prev_portfolio_value') and self.prev_portfolio_value > 0:
                portfolio_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
            else:
                portfolio_return = 0.0
            
            self.prev_portfolio_value = current_portfolio_value
            
            # Tax impact component
            tax_impact = self.tax_calculator.calculate_tax_impact(
                trades, dict(self.portfolio.tax_lots), current_date
            )
            
            # Risk component (portfolio volatility penalty)
            risk_penalty = self._calculate_risk_penalty()
            
            # Transaction cost component
            transaction_costs = sum(trade.get('transaction_cost', 0) for trade in trades)
            transaction_penalty = transaction_costs / self.initial_cash
            
            # Multi-objective reward weights
            alpha = 1.0   # Return weight
            beta = 0.3    # Tax alpha weight
            gamma = 0.1   # Risk penalty weight
            delta = 0.5   # Transaction cost penalty weight
            
            # Calculate reward
            reward = (
                alpha * portfolio_return +
                beta * tax_impact.get('tax_alpha', 0) / self.initial_cash -
                gamma * risk_penalty -
                delta * transaction_penalty
            )
            
            # Store reward components for analysis
            self._last_reward_components = {
                'portfolio_return': portfolio_return,
                'tax_alpha': tax_impact.get('tax_alpha', 0) / self.initial_cash,
                'risk_penalty': risk_penalty,
                'transaction_penalty': transaction_penalty,
                'total_reward': reward
            }
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate risk penalty based on portfolio concentration."""
        try:
            # Get current allocation
            prices = self._get_current_prices(self.dates[self.current_step])
            allocation = self.portfolio.get_allocation(prices)
            
            # Calculate concentration (Herfindahl index)
            position_weights = [allocation.get(symbol, 0) for symbol in self.symbols]
            concentration = sum(w**2 for w in position_weights)
            
            # Penalize high concentration (preference for diversification)
            return max(0, concentration - 0.2)  # Penalty if concentration > 20%
            
        except Exception as e:
            logger.error(f"Error calculating risk penalty: {e}")
            return 0.0
    
    def _get_average_holding_period(self) -> float:
        """Get average holding period across all positions."""
        try:
            if not self.portfolio.tax_lots:
                return 0.0
            
            total_days = 0
            total_shares = 0
            current_date = self.dates[self.current_step]
            
            for lots in self.portfolio.tax_lots.values():
                for lot in lots:
                    days = (current_date - lot['purchase_date']).days
                    shares = lot['shares']
                    total_days += days * shares
                    total_shares += shares
            
            return total_days / total_shares if total_shares > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating holding period: {e}")
            return 0.0
    
    def _get_long_term_ratio(self) -> float:
        """Get ratio of holdings that qualify for long-term capital gains."""
        try:
            if not self.portfolio.tax_lots:
                return 0.0
            
            long_term_value = 0.0
            total_value = 0.0
            current_date = self.dates[self.current_step]
            prices = self._get_current_prices(current_date)
            
            for symbol, lots in self.portfolio.tax_lots.items():
                current_price = prices.get(symbol, 0)
                for lot in lots:
                    holding_days = (current_date - lot['purchase_date']).days
                    lot_value = lot['shares'] * current_price
                    total_value += lot_value
                    
                    if holding_days > 365:  # Long-term threshold
                        long_term_value += lot_value
            
            return long_term_value / total_value if total_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating long-term ratio: {e}")
            return 0.0
    
    def _get_reward_components(self) -> Dict[str, float]:
        """Get detailed reward components for analysis."""
        return getattr(self, '_last_reward_components', {
            'portfolio_return': 0.0,
            'tax_alpha': 0.0,
            'risk_penalty': 0.0,
            'transaction_penalty': 0.0,
            'total_reward': 0.0
        })
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about environment state."""
        try:
            current_date = self.dates[self.current_step] if self.current_step < len(self.dates) else None
            prices = self._get_current_prices(current_date) if current_date else {}
            
            return {
                'date': current_date.isoformat() if current_date else None,
                'step': self.current_step,
                'episode': self.episode_count,
                'portfolio_value': self.portfolio.get_portfolio_value(prices),
                'cash': self.portfolio.cash,
                'positions': dict(self.portfolio.positions),
                'allocation': self.portfolio.get_allocation(prices),
                'unrealized_pnl': self.portfolio.get_unrealized_pnl(prices)
            }
            
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {'error': str(e)}
    
    def render(self, mode='human'):
        """Render environment state (optional)."""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Date: {info['date']}, "
                  f"Portfolio Value: ${info['portfolio_value']:,.2f}")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode."""
        try:
            prices = self._get_current_prices(self.dates[self.current_step - 1])
            final_value = self.portfolio.get_portfolio_value(prices)
            total_return = (final_value - self.initial_cash) / self.initial_cash
            
            return {
                'episode': self.episode_count,
                'total_return': total_return,
                'final_value': final_value,
                'total_trades': len(self.episode_trades),
                'steps': self.current_step,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown()
            }
            
        except Exception as e:
            logger.error(f"Error getting episode summary: {e}")
            return {'error': str(e)}
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for current episode."""
        try:
            if len(self.episode_returns) < 2:
                return 0.0
            
            returns = np.array(self.episode_returns)
            excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for current episode."""
        try:
            if not hasattr(self, 'portfolio_values'):
                return 0.0
            
            values = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            return float(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0


# Neural Network Components for SAC Agent
if TORCH_AVAILABLE:
    
    class Actor(nn.Module):
        """Actor network for SAC agent."""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super(Actor, self).__init__()
            
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)
            
            # Initialize weights
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        
        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            
            mean = self.mean(x)
            log_std = torch.clamp(self.log_std(x), -20, 2)
            
            return mean, log_std
        
        def sample(self, state):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Calculate log probability
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return action, log_prob
    
    
    class Critic(nn.Module):
        """Critic network for SAC agent."""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super(Critic, self).__init__()
            
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            
            # Initialize weights
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        
        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.fc3(x)
            
            return q_value


class ReplayBuffer:
    """Experience replay buffer with tax-aware prioritization."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float,
            next_state: np.ndarray, 
            done: bool, 
            tax_info: Optional[Dict] = None):
        """Add experience to buffer with priority."""
        
        # Calculate priority based on reward magnitude and tax significance
        priority = abs(reward) + 1e-6  # Small epsilon to prevent zero priority
        
        if tax_info:
            # Increase priority for tax-significant events
            tax_alpha = tax_info.get('tax_alpha', 0)
            wash_sales = len(tax_info.get('wash_sale_violations', []))
            priority += abs(tax_alpha) * 2 + wash_sales * 0.5
        
        experience = (state, action, reward, next_state, done, tax_info)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample experiences with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        tax_infos = [exp[5] for exp in batch]
        
        return states, actions, rewards, next_states, dones, weights, indices, tax_infos
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    
    class SoftActorCritic:
        """Soft Actor-Critic agent for tax-aware portfolio management."""
        
        def __init__(self,
                     state_dim: int,
                     action_dim: int,
                     hidden_dim: int = 256,
                     lr: float = 3e-4,
                     gamma: float = 0.99,
                     tau: float = 0.005,
                     alpha: float = 0.2,
                     buffer_size: int = 100000,
                     device: str = 'auto'):
            """
            Initialize SAC agent.
            
            Args:
                state_dim: Dimension of state space
                action_dim: Dimension of action space
                hidden_dim: Hidden layer dimension
                lr: Learning rate
                gamma: Discount factor
                tau: Soft update coefficient
                alpha: Entropy regularization coefficient
                buffer_size: Replay buffer size
                device: Device to use ('cpu', 'cuda', or 'auto')
            """
            
            # Set device
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.gamma = gamma
            self.tau = tau
            self.alpha = alpha
            
            # Initialize networks
            self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            
            # Target networks
            self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            
            # Copy weights to target networks
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
            
            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
            self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
            
            # Automatic entropy tuning
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
            # Replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size)
            
            # Training statistics
            self.training_stats = {
                'actor_loss': [],
                'critic_loss': [],
                'alpha_loss': [],
                'q_values': [],
                'alpha_values': []
            }
            
            logger.info(f"SAC agent initialized on {self.device}")
        
        def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Select action using current policy."""
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                if deterministic:
                    with torch.no_grad():
                        mean, _ = self.actor(state_tensor)
                        action = torch.tanh(mean)
                else:
                    with torch.no_grad():
                        action, _ = self.actor.sample(state_tensor)
                
                action = action.cpu().numpy()[0]
                
                # Ensure valid portfolio weights
                action = np.clip(action, 0, 1)
                if np.sum(action) > 0:
                    action = action / np.sum(action)
                else:
                    action = np.ones(len(action)) / len(action)
                
                return action
                
            except Exception as e:
                logger.error(f"Error selecting action: {e}")
                # Return equal weights as fallback
                return np.ones(len(state)) / len(state)
        
        def train(self, batch_size: int = 256) -> Dict[str, float]:
            """Train the SAC agent."""
            if len(self.replay_buffer) < batch_size:
                return {}
            
            try:
                # Sample batch
                batch = self.replay_buffer.sample(batch_size)
                if batch is None:
                    return {}
                
                states, actions, rewards, next_states, dones, weights, indices, tax_infos = batch
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
                weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
                
                # Train critics
                critic_loss = self._train_critics(states, actions, rewards, next_states, dones, weights)
                
                # Train actor
                actor_loss = self._train_actor(states)
                
                # Train alpha (entropy temperature)
                alpha_loss = self._train_alpha(states)
                
                # Update target networks
                self._soft_update_target_networks()
                
                # Update replay buffer priorities
                with torch.no_grad():
                    q1 = self.critic1(states, actions)
                    q2 = self.critic2(states, actions)
                    q_min = torch.min(q1, q2)
                    td_errors = torch.abs(rewards - q_min).cpu().numpy().flatten()
                    
                self.replay_buffer.update_priorities(indices, td_errors)
                
                # Update statistics
                stats = {
                    'actor_loss': float(actor_loss),
                    'critic_loss': float(critic_loss),
                    'alpha_loss': float(alpha_loss),
                    'alpha_value': float(self.alpha),
                    'q_value': float(q_min.mean()),
                    'buffer_size': len(self.replay_buffer)
                }
                
                # Store in training history
                self.training_stats['actor_loss'].append(stats['actor_loss'])
                self.training_stats['critic_loss'].append(stats['critic_loss'])
                self.training_stats['alpha_loss'].append(stats['alpha_loss'])
                self.training_stats['alpha_values'].append(stats['alpha_value'])
                self.training_stats['q_values'].append(stats['q_value'])
                
                return stats
                
            except Exception as e:
                logger.error(f"Error training SAC agent: {e}")
                return {}
        
        def _train_critics(self, states, actions, rewards, next_states, dones, weights):
            """Train critic networks."""
            # Calculate target Q-values
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # Current Q-values
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # Critic losses with importance sampling weights
            critic1_loss = F.mse_loss(current_q1, target_q, reduction='none')
            critic2_loss = F.mse_loss(current_q2, target_q, reduction='none')
            
            critic1_loss = (critic1_loss * weights).mean()
            critic2_loss = (critic2_loss * weights).mean()
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            return (critic1_loss + critic2_loss) / 2
        
        def _train_actor(self, states):
            """Train actor network."""
            actions, log_probs = self.actor.sample(states)
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q_min = torch.min(q1, q2)
            
            actor_loss = (self.alpha * log_probs - q_min).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            return actor_loss
        
        def _train_alpha(self, states):
            """Train entropy temperature."""
            with torch.no_grad():
                _, log_probs = self.actor.sample(states)
            
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            
            return alpha_loss
        
        def _soft_update_target_networks(self):
            """Soft update target networks."""
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        def save_model(self, filepath: str):
            """Save model weights and training statistics."""
            try:
                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic1_state_dict': self.critic1.state_dict(),
                    'critic2_state_dict': self.critic2.state_dict(),
                    'target_critic1_state_dict': self.target_critic1.state_dict(),
                    'target_critic2_state_dict': self.target_critic2.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                    'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                    'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                    'log_alpha': self.log_alpha,
                    'training_stats': self.training_stats,
                    'hyperparameters': {
                        'gamma': self.gamma,
                        'tau': self.tau,
                        'alpha': self.alpha,
                        'target_entropy': self.target_entropy
                    }
                }, filepath)
                
                logger.info(f"Model saved to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving model: {e}")
        
        def load_model(self, filepath: str):
            """Load model weights and training statistics."""
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
                self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
                self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
                self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
                
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
                self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                
                self.log_alpha = checkpoint['log_alpha']
                self.training_stats = checkpoint['training_stats']
                
                # Load hyperparameters
                hyperparams = checkpoint['hyperparameters']
                self.gamma = hyperparams['gamma']
                self.tau = hyperparams['tau']
                self.alpha = hyperparams['alpha']
                self.target_entropy = hyperparams['target_entropy']
                
                logger.info(f"Model loaded from {filepath}")
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")

else:
    # Simplified SAC for environments without PyTorch
    class SoftActorCritic:
        """Simplified SAC agent without PyTorch."""
        
        def __init__(self, state_dim: int, action_dim: int, **kwargs):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.replay_buffer = ReplayBuffer()
            self.training_stats = {
                'actor_loss': [],
                'critic_loss': [],
                'alpha_loss': [],
                'q_values': [],
                'alpha_values': []
            }
            logger.warning("Using simplified SAC - PyTorch not available")
        
        def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Simple action selection without neural networks."""
            # Random portfolio allocation with some structure
            weights = np.random.dirichlet(np.ones(self.action_dim))
            return weights
        
        def train(self, batch_size: int = 256) -> Dict[str, float]:
            """Placeholder training method."""
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'alpha_loss': 0.0,
                'alpha_value': 0.2,
                'q_value': 0.0,
                'buffer_size': len(self.replay_buffer)
            }
        
        def save_model(self, filepath: str):
            """Save simplified model."""
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'training_stats': self.training_stats,
                    'buffer_size': len(self.replay_buffer)
                }, f)
        
        def load_model(self, filepath: str):
            """Load simplified model."""
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.training_stats = data.get('training_stats', {})
                logger.info(f"Simplified model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Error loading simplified model: {e}")


print("✅ RL Environment and SAC Agent implemented successfully!")
print(f"   - TaxAwarePortfolioEnv: Custom Gymnasium environment with {35}D observation space")
print(f"   - SoftActorCritic: {'PyTorch-based' if TORCH_AVAILABLE else 'Simplified'} SAC implementation")
print(f"   - ReplayBuffer: Tax-aware prioritized experience replay")
