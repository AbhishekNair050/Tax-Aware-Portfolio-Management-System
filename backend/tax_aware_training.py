"""
Tax-Aware Portfolio Management System - Training Manager and Curriculum Learning
This module contains the training orchestration and curriculum learning components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime as dt
import logging
import json
import os
import time
from dataclasses import dataclass
from enum import Enum

from tax_aware_core import TaxCalculator, PortfolioState, MarketDataManager
from tax_aware_rl import TaxAwarePortfolioEnv, SoftActorCritic, TORCH_AVAILABLE

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages."""
    BASIC_TRADING = "basic_trading"
    SIMPLE_TAX = "simple_tax"
    FULL_COMPLEXITY = "full_complexity"
    ADVANCED_STRATEGIES = "advanced_strategies"


@dataclass
class CurriculumConfig:
    """Configuration for a curriculum stage."""
    stage: CurriculumStage
    symbols: List[str]
    episodes: int
    success_threshold: float
    tax_complexity: str
    transaction_cost: float
    max_position_size: float
    description: str


class CurriculumManager:
    """Manages curriculum learning progression with adaptive thresholds."""
    
    def __init__(self, custom_stages: Optional[List[CurriculumConfig]] = None):
        """
        Initialize curriculum manager.
        
        Args:
            custom_stages: Custom curriculum stages (uses default if None)
        """
        if custom_stages:
            self.stages = {stage.stage: stage for stage in custom_stages}
        else:
            self.stages = self._create_default_curriculum()
        
        self.current_stage = CurriculumStage.BASIC_TRADING
        self.stage_progress = {
            'episodes_completed': 0,
            'success_episodes': 0,
            'average_reward': 0.0,
            'best_reward': float('-inf'),
            'recent_rewards': []
        }
        
        self.curriculum_history = []
        
        logger.info(f"Curriculum initialized with {len(self.stages)} stages")
    
    def _create_default_curriculum(self) -> Dict[CurriculumStage, CurriculumConfig]:
        """Create default curriculum stages."""
        return {
            CurriculumStage.BASIC_TRADING: CurriculumConfig(
                stage=CurriculumStage.BASIC_TRADING,
                symbols=['AAPL', 'MSFT'],
                episodes=500,
                success_threshold=0.05,  # 5% avg return threshold
                tax_complexity='none',
                transaction_cost=0.001,
                max_position_size=0.6,
                description="Learn basic portfolio allocation with 2 assets"
            ),
            CurriculumStage.SIMPLE_TAX: CurriculumConfig(
                stage=CurriculumStage.SIMPLE_TAX,
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                episodes=1000,
                success_threshold=0.03,  # 3% avg return threshold
                tax_complexity='capital_gains_only',
                transaction_cost=0.0015,
                max_position_size=0.5,
                description="Introduce basic tax considerations with 3 assets"
            ),
            CurriculumStage.FULL_COMPLEXITY: CurriculumConfig(
                stage=CurriculumStage.FULL_COMPLEXITY,
                symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
                episodes=2000,
                success_threshold=0.02,  # 2% avg return threshold
                tax_complexity='full_tax_with_wash_sales',
                transaction_cost=0.002,
                max_position_size=0.4,
                description="Full tax-aware optimization with 5 assets"
            ),
            CurriculumStage.ADVANCED_STRATEGIES: CurriculumConfig(
                stage=CurriculumStage.ADVANCED_STRATEGIES,
                symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META'],
                episodes=3000,
                success_threshold=0.015,  # 1.5% avg return threshold
                tax_complexity='advanced_tax_strategies',
                transaction_cost=0.0025,
                max_position_size=0.3,
                description="Advanced tax strategies with 7 assets"
            )
        }
    
    def update_progress(self, episode_reward: float, episode_info: Dict[str, Any]) -> None:
        """Update progress for current stage."""
        self.stage_progress['episodes_completed'] += 1
        self.stage_progress['recent_rewards'].append(episode_reward)
        
        # Keep only last 50 episodes for moving average
        if len(self.stage_progress['recent_rewards']) > 50:
            self.stage_progress['recent_rewards'].pop(0)
        
        # Update statistics
        self.stage_progress['average_reward'] = np.mean(self.stage_progress['recent_rewards'])
        self.stage_progress['best_reward'] = max(self.stage_progress['best_reward'], episode_reward)
        
        # Count success episodes (above threshold)
        current_config = self.stages[self.current_stage]
        if episode_reward > current_config.success_threshold:
            self.stage_progress['success_episodes'] += 1
    
    def should_advance_stage(self) -> bool:
        """Check if agent should advance to next curriculum stage."""
        current_config = self.stages[self.current_stage]
        
        # Minimum episodes requirement
        if self.stage_progress['episodes_completed'] < current_config.episodes:
            return False
        
        # Performance requirements
        min_episodes_for_evaluation = 20
        if len(self.stage_progress['recent_rewards']) < min_episodes_for_evaluation:
            return False
        
        # Check average performance over recent episodes
        recent_avg = np.mean(self.stage_progress['recent_rewards'][-min_episodes_for_evaluation:])
        performance_met = recent_avg > current_config.success_threshold
        
        # Check consistency (avoid advancing on lucky streaks)
        success_rate = self.stage_progress['success_episodes'] / self.stage_progress['episodes_completed']
        consistency_met = success_rate > 0.4  # At least 40% success rate
        
        return performance_met and consistency_met
    
    def advance_stage(self) -> Optional[CurriculumStage]:
        """Advance to next curriculum stage."""
        # Record current stage completion
        stage_summary = {
            'stage': self.current_stage.value,
            'episodes_completed': self.stage_progress['episodes_completed'],
            'average_reward': self.stage_progress['average_reward'],
            'best_reward': self.stage_progress['best_reward'],
            'success_rate': self.stage_progress['success_episodes'] / self.stage_progress['episodes_completed'],
            'completion_time': dt.datetime.now().isoformat()
        }
        self.curriculum_history.append(stage_summary)
        
        # Find next stage
        stage_order = list(CurriculumStage)
        current_idx = stage_order.index(self.current_stage)
        
        if current_idx < len(stage_order) - 1:
            self.current_stage = stage_order[current_idx + 1]
            
            # Reset progress for new stage
            self.stage_progress = {
                'episodes_completed': 0,
                'success_episodes': 0,
                'average_reward': 0.0,
                'best_reward': float('-inf'),
                'recent_rewards': []
            }
            
            logger.info(f"Advanced to curriculum stage: {self.current_stage.value}")
            return self.current_stage
        
        logger.info("Curriculum completed - all stages mastered")
        return None
    
    def get_current_config(self) -> CurriculumConfig:
        """Get current stage configuration."""
        return self.stages[self.current_stage]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        current_config = self.get_current_config()
        
        return {
            'current_stage': self.current_stage.value,
            'stage_description': current_config.description,
            'progress': self.stage_progress,
            'stage_config': {
                'symbols': current_config.symbols,
                'target_episodes': current_config.episodes,
                'success_threshold': current_config.success_threshold,
                'tax_complexity': current_config.tax_complexity
            },
            'curriculum_history': self.curriculum_history,
            'completion_percentage': self._calculate_completion_percentage()
        }
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate overall curriculum completion percentage."""
        total_stages = len(self.stages)
        completed_stages = len(self.curriculum_history)
        
        # Add partial progress for current stage
        current_config = self.get_current_config()
        current_progress = min(1.0, self.stage_progress['episodes_completed'] / current_config.episodes)
        
        return ((completed_stages + current_progress) / total_stages) * 100


class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    
    def __init__(self):
        self.episode_metrics = []
        self.training_start_time = None
        self.training_end_time = None
        
    def record_episode(self, 
                      episode: int,
                      reward: float,
                      portfolio_value: float,
                      total_return: float,
                      trades: List[Dict],
                      tax_info: Dict[str, Any],
                      agent_stats: Dict[str, float],
                      curriculum_stage: str) -> None:
        """Record metrics for completed episode."""
        
        episode_data = {
            'episode': episode,
            'timestamp': dt.datetime.now().isoformat(),
            'reward': reward,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'tax_alpha': tax_info.get('tax_alpha', 0.0),
            'tax_liability': tax_info.get('total_tax_liability', 0.0),
            'curriculum_stage': curriculum_stage,
            'agent_stats': agent_stats
        }
        
        self.episode_metrics.append(episode_data)
    
    def get_performance_summary(self, last_n_episodes: int = 100) -> Dict[str, Any]:
        """Get performance summary for last N episodes."""
        if not self.episode_metrics:
            return {}
        
        recent_episodes = self.episode_metrics[-last_n_episodes:]
        
        rewards = [ep['reward'] for ep in recent_episodes]
        returns = [ep['total_return'] for ep in recent_episodes]
        portfolio_values = [ep['portfolio_value'] for ep in recent_episodes]
        
        return {
            'episodes_analyzed': len(recent_episodes),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'average_return': np.mean(returns),
            'return_std': np.std(returns),
            'best_episode': max(recent_episodes, key=lambda x: x['reward']),
            'worst_episode': min(recent_episodes, key=lambda x: x['reward']),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'average_trades_per_episode': np.mean([ep['num_trades'] for ep in recent_episodes])
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(excess_returns))
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(np.min(drawdown))
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        try:
            export_data = {
                'training_summary': {
                    'start_time': self.training_start_time,
                    'end_time': self.training_end_time,
                    'total_episodes': len(self.episode_metrics),
                    'total_duration_hours': self._calculate_training_duration()
                },
                'performance_summary': self.get_performance_summary(),
                'episode_metrics': self.episode_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Training metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def _calculate_training_duration(self) -> float:
        """Calculate training duration in hours."""
        if self.training_start_time and self.training_end_time:
            start = dt.datetime.fromisoformat(self.training_start_time)
            end = dt.datetime.fromisoformat(self.training_end_time)
            return (end - start).total_seconds() / 3600
        return 0.0


class TaxAwareRLTrainer:
    """Main training manager for tax-aware portfolio RL with curriculum learning."""
    
    def __init__(self,
                 symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                 initial_cash: float = 1000000.0,
                 train_start: str = '2020-01-01',
                 train_end: str = '2022-12-31',
                 val_start: str = '2023-01-01',
                 val_end: str = '2023-12-31',
                 use_curriculum: bool = True,
                 save_dir: str = './models',
                 device: str = 'auto'):
        """
        Initialize the training manager.
        
        Args:
            symbols: List of trading symbols
            initial_cash: Starting portfolio value
            train_start: Training period start date
            train_end: Training period end date
            val_start: Validation period start date
            val_end: Validation period end date
            use_curriculum: Whether to use curriculum learning
            save_dir: Directory to save models and logs
            device: Device for training ('cpu', 'cuda', or 'auto')
        """
        
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.use_curriculum = use_curriculum
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize curriculum manager
        if use_curriculum:
            self.curriculum = CurriculumManager()
        else:
            self.curriculum = None
        
        # Initialize environments
        self._initialize_environments()
        
        # Initialize agent
        if hasattr(self, 'train_env'):
            state_dim = self.train_env.observation_space.shape[0]
            action_dim = self.train_env.action_space.shape[0]
            self.agent = SoftActorCritic(state_dim, action_dim, device=device)
        else:
            logger.error("Failed to initialize training environment")
            return
        
        # Initialize metrics tracking
        self.metrics = TrainingMetrics()
        
        # Training state
        self.total_episodes = 0
        self.is_training = False
        self.training_interrupted = False
        
        logger.info(f"Trainer initialized with {len(symbols)} symbols, curriculum: {use_curriculum}")
    
    def _initialize_environments(self):
        """Initialize training and validation environments."""
        try:
            # Get initial curriculum configuration
            if self.use_curriculum:
                config = self.curriculum.get_current_config()
                env_symbols = config.symbols
                transaction_cost = config.transaction_cost
                max_position_size = config.max_position_size
            else:
                env_symbols = self.symbols
                transaction_cost = 0.002
                max_position_size = 0.4
            
            # Training environment
            self.train_env = TaxAwarePortfolioEnv(
                symbols=env_symbols,
                initial_cash=self.initial_cash,
                start_date=self.train_start,
                end_date=self.train_end,
                transaction_cost=transaction_cost,
                max_position_size=max_position_size
            )
            
            # Validation environment
            self.val_env = TaxAwarePortfolioEnv(
                symbols=self.symbols,  # Always use full symbol set for validation
                initial_cash=self.initial_cash,
                start_date=self.val_start,
                end_date=self.val_end,
                transaction_cost=0.002,
                max_position_size=0.4
            )
            
            logger.info(f"Environments initialized: {len(env_symbols)} training symbols")
            
        except Exception as e:
            logger.error(f"Error initializing environments: {e}")
    
    def _update_environment_for_curriculum(self):
        """Update training environment for current curriculum stage."""
        if not self.use_curriculum:
            return
        
        try:
            config = self.curriculum.get_current_config()
            
            # Create new environment with curriculum configuration
            self.train_env = TaxAwarePortfolioEnv(
                symbols=config.symbols,
                initial_cash=self.initial_cash,
                start_date=self.train_start,
                end_date=self.train_end,
                transaction_cost=config.transaction_cost,
                max_position_size=config.max_position_size
            )
            
            logger.info(f"Environment updated for stage: {config.stage.value}")
            
        except Exception as e:
            logger.error(f"Error updating environment for curriculum: {e}")
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode."""
        try:
            state, info = self.train_env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_trades = []
            
            done = False
            while not done and not self.training_interrupted:
                # Select action
                action = self.agent.select_action(state)
                
                # Execute step
                next_state, reward, terminated, truncated, step_info = self.train_env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.agent.replay_buffer.add(
                    state, action, reward, next_state, done, 
                    step_info.get('tax_info', None)
                )
                
                # Train agent
                if len(self.agent.replay_buffer) > 256:
                    agent_stats = self.agent.train()
                else:
                    agent_stats = {}
                
                # Track episode data
                episode_reward += reward
                episode_steps += 1
                episode_trades.extend(step_info.get('trades', []))
                
                state = next_state
            
            # Calculate final metrics
            final_portfolio_value = step_info.get('portfolio_value', self.initial_cash)
            total_return = (final_portfolio_value - self.initial_cash) / self.initial_cash
            
            # Get tax information
            tax_info = {}
            if episode_trades:
                tax_info = self.train_env.tax_calculator.calculate_tax_impact(
                    episode_trades, 
                    dict(self.train_env.portfolio.tax_lots),
                    self.train_env.dates[self.train_env.current_step - 1]
                )
            
            # Record episode metrics
            curriculum_stage = self.curriculum.current_stage.value if self.curriculum else "no_curriculum"
            self.metrics.record_episode(
                episode=self.total_episodes + 1,
                reward=episode_reward,
                portfolio_value=final_portfolio_value,
                total_return=total_return,
                trades=episode_trades,
                tax_info=tax_info,
                agent_stats=agent_stats,
                curriculum_stage=curriculum_stage
            )
            
            # Update curriculum progress
            if self.curriculum:
                self.curriculum.update_progress(episode_reward, step_info)
            
            self.total_episodes += 1
            
            return {
                'episode': self.total_episodes,
                'reward': episode_reward,
                'steps': episode_steps,
                'portfolio_value': final_portfolio_value,
                'total_return': total_return,
                'num_trades': len(episode_trades),
                'tax_alpha': tax_info.get('tax_alpha', 0.0),
                'tax_liability': tax_info.get('total_tax_liability', 0.0),
                'agent_stats': agent_stats,
                'curriculum_stage': curriculum_stage
            }
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return {'error': str(e)}
    
    def validate_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Validate agent performance on validation set."""
        try:
            validation_results = []
            
            for episode in range(num_episodes):
                state, info = self.val_env.reset()
                episode_reward = 0.0
                episode_trades = []
                
                done = False
                while not done:
                    # Use deterministic policy for validation
                    action = self.agent.select_action(state, deterministic=True)
                    state, reward, terminated, truncated, step_info = self.val_env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_trades.extend(step_info.get('trades', []))
                
                # Calculate metrics
                final_value = step_info.get('portfolio_value', self.initial_cash)
                total_return = (final_value - self.initial_cash) / self.initial_cash
                
                validation_results.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'portfolio_value': final_value,
                    'total_return': total_return,
                    'num_trades': len(episode_trades)
                })
            
            # Calculate validation statistics
            rewards = [r['reward'] for r in validation_results]
            returns = [r['total_return'] for r in validation_results]
            values = [r['portfolio_value'] for r in validation_results]
            
            validation_summary = {
                'num_episodes': num_episodes,
                'average_reward': np.mean(rewards),
                'reward_std': np.std(rewards),
                'average_return': np.mean(returns),
                'return_std': np.std(returns),
                'average_final_value': np.mean(values),
                'best_return': max(returns),
                'worst_return': min(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'episodes': validation_results
            }
            
            logger.info(f"Validation completed: Avg Return={validation_summary['average_return']:.4f}")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {'error': str(e)}
    
    def train(self, 
              num_episodes: int = 1000,
              validate_every: int = 100,
              save_every: int = 500,
              early_stopping_patience: int = 200) -> Dict[str, Any]:
        """
        Main training loop with curriculum learning and validation.
        
        Args:
            num_episodes: Total number of episodes to train
            validate_every: Frequency of validation runs
            save_every: Frequency of model saves
            early_stopping_patience: Episodes to wait for improvement before stopping
            
        Returns:
            Training summary dictionary
        """
        
        logger.info(f"Starting training for {num_episodes} episodes")
        self.metrics.training_start_time = dt.datetime.now().isoformat()
        self.is_training = True
        self.training_interrupted = False
        
        best_validation_score = float('-inf')
        episodes_without_improvement = 0
        
        try:
            for episode in range(num_episodes):
                if self.training_interrupted:
                    logger.info("Training interrupted by user")
                    break
                
                # Train one episode
                episode_result = self.train_episode()
                
                if 'error' in episode_result:
                    logger.error(f"Episode {episode + 1} failed: {episode_result['error']}")
                    continue
                
                # Print progress
                if (episode + 1) % 50 == 0:
                    logger.info(f"Episode {episode + 1}/{num_episodes}: "
                              f"Reward={episode_result['reward']:.4f}, "
                              f"Return={episode_result['total_return']:.4f}, "
                              f"Portfolio=${episode_result['portfolio_value']:,.0f}")
                
                # Validation
                if (episode + 1) % validate_every == 0:
                    validation_result = self.validate_agent()
                    
                    if 'error' not in validation_result:
                        current_score = validation_result['average_return']
                        
                        if current_score > best_validation_score:
                            best_validation_score = current_score
                            episodes_without_improvement = 0
                            
                            # Save best model
                            self.save_model(f"{self.save_dir}/best_model.pth")
                            logger.info(f"New best validation score: {current_score:.4f}")
                        else:
                            episodes_without_improvement += validate_every
                        
                        logger.info(f"Validation - Avg Return: {current_score:.4f}, "
                                  f"Win Rate: {validation_result['win_rate']:.2%}")
                
                # Save checkpoint
                if (episode + 1) % save_every == 0:
                    self.save_model(f"{self.save_dir}/checkpoint_episode_{episode + 1}.pth")
                    self.metrics.export_metrics(f"{self.save_dir}/training_metrics_{episode + 1}.json")
                
                # Check curriculum advancement
                if self.curriculum and self.curriculum.should_advance_stage():
                    new_stage = self.curriculum.advance_stage()
                    if new_stage:
                        self._update_environment_for_curriculum()
                        logger.info(f"Curriculum advanced to: {new_stage.value}")
                
                # Early stopping check
                if episodes_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {episodes_without_improvement} episodes without improvement")
                    break
            
            # Final validation
            final_validation = self.validate_agent(num_episodes=20)
            
            # Training completed
            self.metrics.training_end_time = dt.datetime.now().isoformat()
            self.is_training = False
            
            # Save final model and metrics
            self.save_model(f"{self.save_dir}/final_model.pth")
            self.metrics.export_metrics(f"{self.save_dir}/final_training_metrics.json")
            
            # Prepare training summary
            training_summary = {
                'total_episodes_trained': self.total_episodes,
                'final_validation': final_validation,
                'best_validation_score': best_validation_score,
                'training_duration_hours': self.metrics._calculate_training_duration(),
                'curriculum_progress': self.curriculum.get_progress_summary() if self.curriculum else None,
                'performance_summary': self.metrics.get_performance_summary(),
                'early_stopped': episodes_without_improvement >= early_stopping_patience
            }
            
            logger.info("Training completed successfully!")
            logger.info(f"Final validation score: {final_validation.get('average_return', 0):.4f}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.is_training = False
            return {'error': str(e)}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        try:
            base_stats = {
                'total_episodes': self.total_episodes,
                'is_training': self.is_training,
                'buffer_size': len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0
            }
            
            # Add curriculum information
            if self.curriculum:
                base_stats['curriculum'] = self.curriculum.get_progress_summary()
            
            # Add recent performance metrics
            if self.metrics.episode_metrics:
                base_stats['performance'] = self.metrics.get_performance_summary(last_n_episodes=50)
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model with metadata."""
        try:
            # Save agent model
            self.agent.save_model(filepath)
            
            # Save additional metadata
            metadata = {
                'symbols': self.symbols,
                'initial_cash': self.initial_cash,
                'total_episodes': self.total_episodes,
                'training_period': f"{self.train_start} to {self.train_end}",
                'curriculum_used': self.use_curriculum,
                'save_timestamp': dt.datetime.now().isoformat()
            }
            
            if self.curriculum:
                metadata['curriculum_progress'] = self.curriculum.get_progress_summary()
            
            metadata_path = filepath.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model and metadata saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model with metadata."""
        try:
            # Load agent model
            self.agent.load_model(filepath)
            
            # Load metadata if available
            metadata_path = filepath.replace('.pth', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.total_episodes = metadata.get('total_episodes', 0)
                logger.info(f"Loaded model with {self.total_episodes} training episodes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def stop_training(self) -> None:
        """Gracefully stop training."""
        self.training_interrupted = True
        logger.info("Training stop requested")


# Example training configuration
def create_sample_trainer():
    """Create a sample trainer for demonstration."""
    try:
        trainer = TaxAwareRLTrainer(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            initial_cash=1000000.0,
            use_curriculum=True,
            save_dir='./tax_aware_models'
        )
        
        logger.info("Sample trainer created successfully")
        return trainer
        
    except Exception as e:
        logger.error(f"Error creating sample trainer: {e}")
        return None


print("✅ Training Manager and Curriculum Learning implemented successfully!")
print("   - CurriculumManager: 4-stage progressive learning")
print("   - TrainingMetrics: Comprehensive performance tracking")
print("   - TaxAwareRLTrainer: Complete training orchestration with validation")
