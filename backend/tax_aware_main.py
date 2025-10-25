"""
Tax-Aware Portfolio Management System - Main Application Entry Point
This module provides the main application interface and example usage.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import datetime as dt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tax_aware_core import TaxCalculator, PortfolioState, MarketDataManager
from tax_aware_rl import TaxAwarePortfolioEnv, SoftActorCritic, TORCH_AVAILABLE
from tax_aware_training import TaxAwareRLTrainer, CurriculumManager
from tax_aware_api import create_api, FASTAPI_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tax_aware_rl.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TaxAwarePortfolioSystem:
    """Main system interface for tax-aware portfolio management."""
    
    def __init__(self):
        """Initialize the system."""
        self.system_info = {
            "name": "Tax-Aware Portfolio Management RL System",
            "version": "2.0.0",
            "initialized_at": dt.datetime.now().isoformat(),
            "components": {
                "tax_calculator": True,
                "portfolio_state": True,
                "market_data": True,
                "rl_environment": True,
                "sac_agent": TORCH_AVAILABLE,
                "training_manager": True,
                "curriculum_learning": True,
                "rest_api": FASTAPI_AVAILABLE
            }
        }
        
        logger.info("Tax-Aware Portfolio Management System initialized")
    
    def print_system_status(self):
        """Print comprehensive system status."""
        print("\n" + "="*80)
        print("🏦 TAX-AWARE PORTFOLIO MANAGEMENT RL SYSTEM")
        print("="*80)
        
        print(f"\n📊 System Information:")
        print(f"   Version: {self.system_info['version']}")
        print(f"   Initialized: {self.system_info['initialized_at']}")
        
        print(f"\n🔧 Component Status:")
        for component, status in self.system_info['components'].items():
            status_icon = "✅" if status else "❌"
            component_name = component.replace('_', ' ').title()
            print(f"   {status_icon} {component_name}")
        
        print(f"\n🚀 Available Features:")
        features = [
            "Multi-Asset Portfolio Optimization",
            "Tax-Loss Harvesting Strategies",
            "Long/Short Term Capital Gains Optimization",
            "Wash Sale Rule Compliance",
            "Real-Time Market Data Integration",
            "Soft Actor-Critic Reinforcement Learning",
            "Curriculum Learning for Progressive Training",
            "Comprehensive Backtesting Framework",
            "Tax Impact Analysis and Reporting"
        ]
        
        if FASTAPI_AVAILABLE:
            features.append("REST API for Frontend Integration")
        
        for feature in features:
            print(f"   • {feature}")
        
        print(f"\n📈 Supported Assets:")
        print(f"   • Equities (US Markets)")
        print(f"   • ETFs and Index Funds")
        print(f"   • Real-time data via YFinance")
        print(f"   • Technical indicators and market analysis")
        
        print(f"\n🧠 AI/ML Capabilities:")
        print(f"   • Soft Actor-Critic (SAC) Algorithm")
        print(f"   • 35-dimensional observation space")
        print(f"   • Continuous action space for portfolio weights")
        print(f"   • Tax-aware reward function")
        print(f"   • Experience replay with prioritization")
        print(f"   • Curriculum learning with 4 stages")
        
        if not TORCH_AVAILABLE:
            print(f"\n⚠️  PyTorch not available - using simplified agent")
            print(f"   Install with: pip install torch torchvision")
        
        if not FASTAPI_AVAILABLE:
            print(f"\n⚠️  FastAPI not available - API disabled")
            print(f"   Install with: pip install fastapi uvicorn")
        
        print("="*80)
    
    def create_demo_trainer(self, symbols: List[str] = None) -> TaxAwareRLTrainer:
        """Create a demo trainer for testing."""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        try:
            # Use recent dates that should have good data availability
            current_year = dt.datetime.now().year
            
            trainer = TaxAwareRLTrainer(
                symbols=symbols,
                initial_cash=1000000.0,
                train_start=f'{current_year-2}-01-01',  # 2 years ago
                train_end=f'{current_year-1}-12-31',    # Last year
                val_start=f'{current_year}-01-01',      # This year
                val_end=f'{current_year}-06-30',        # Mid this year
                use_curriculum=True,
                save_dir='./demo_models'
            )
            
            logger.info(f"Demo trainer created with {len(symbols)} symbols")
            return trainer
            
        except Exception as e:
            logger.error(f"Error creating demo trainer: {e}")
            return None
    
    def run_quick_demo(self):
        """Run a quick demonstration of the system."""
        print("\n🎯 Running Quick Demo...")
        
        try:
            # 1. Create trainer
            print("\n1. Creating trainer...")
            trainer = self.create_demo_trainer(['AAPL', 'MSFT'])
            
            if not trainer:
                print("❌ Failed to create trainer")
                return
            
            print("✅ Trainer created successfully")
            
            # 2. Train a few episodes
            print("\n2. Training sample episodes...")
            results = []
            
            for i in range(3):
                result = trainer.train_episode()
                if 'error' not in result:
                    results.append(result)
                    print(f"   Episode {i+1}: Reward={result['reward']:.4f}, "
                          f"Return={result['total_return']:.2%}")
                else:
                    print(f"   Episode {i+1}: Error - {result['error']}")
            
            # 3. Get training stats
            print("\n3. Training statistics...")
            stats = trainer.get_training_stats()
            print(f"   Total Episodes: {stats.get('total_episodes', 0)}")
            print(f"   Buffer Size: {stats.get('buffer_size', 0)}")
            
            if 'curriculum' in stats:
                curriculum = stats['curriculum']
                print(f"   Curriculum Stage: {curriculum.get('current_stage', 'N/A')}")
                print(f"   Stage Progress: {curriculum.get('completion_percentage', 0):.1f}%")
            
            # 4. Portfolio analysis demo
            print("\n4. Portfolio analysis demo...")
            market_data = MarketDataManager(['AAPL', 'MSFT'])
            
            # Use recent dates for better data availability
            current_year = dt.datetime.now().year
            start_date = f'{current_year-1}-01-01'
            end_date = f'{current_year-1}-12-31'
            
            print(f"   Fetching data from {start_date} to {end_date}...")
            data = market_data.fetch_data(start_date, end_date)
            
            if not data:
                print("   ❌ Failed to fetch real market data")
                print("   Please check internet connection and try again")
                return
            
            for symbol in ['AAPL', 'MSFT']:
                if symbol in data and not data[symbol].empty:
                    df = data[symbol]
                    if len(df) > 1:
                        total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                        print(f"   ✅ {symbol}: {total_return:.1f}% return ({len(df)} days of real data)")
                    else:
                        print(f"   ⚠️  {symbol}: Limited data available")
                else:
                    print(f"   ❌ {symbol}: No data available")
            
            # 5. Tax calculation demo
            print("\n5. Tax impact calculation...")
            tax_calc = TaxCalculator()
            
            # Simulate a trade
            dummy_trades = [{
                'symbol': 'AAPL',
                'quantity': -100,
                'price': 180.0,
                'date': dt.datetime.now()
            }]
            
            dummy_lots = {
                'AAPL': [{
                    'shares': 200,
                    'cost_basis': 150.0,
                    'purchase_date': dt.datetime(2023, 1, 1)
                }]
            }
            
            tax_impact = tax_calc.calculate_tax_impact(
                dummy_trades, dummy_lots, dt.datetime.now()
            )
            
            print(f"   Tax Liability: ${tax_impact['total_tax_liability']:.2f}")
            print(f"   Tax Alpha: ${tax_impact['tax_alpha']:.2f}")
            
            print("\n✅ Quick demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the REST API server."""
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
            return
        
        try:
            api = create_api(host=host, port=port)
            api.run_server()
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            print(f"❌ Failed to start API server: {e}")
    
    def run_full_training_example(self):
        """Run a comprehensive training example."""
        print("\n🚀 Running Full Training Example...")
        
        try:
            # Create trainer with full configuration
            trainer = TaxAwareRLTrainer(
                symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
                initial_cash=1000000.0,
                train_start='2022-01-01',
                train_end='2023-12-31',
                val_start='2024-01-01',
                val_end='2024-06-30',
                use_curriculum=True,
                save_dir='./full_training_models'
            )
            
            print("✅ Trainer initialized with curriculum learning")
            
            # Run training with validation
            print("\n📈 Starting training (limited episodes for demo)...")
            training_result = trainer.train(
                num_episodes=50,  # Limited for demo
                validate_every=20,
                save_every=25
            )
            
            print(f"\n📊 Training Results:")
            if 'error' not in training_result:
                print(f"   Episodes Trained: {training_result.get('total_episodes_trained', 0)}")
                
                if 'final_validation' in training_result:
                    val = training_result['final_validation']
                    print(f"   Final Validation Return: {val.get('average_return', 0):.2%}")
                    print(f"   Win Rate: {val.get('win_rate', 0):.1%}")
                
                if 'curriculum_progress' in training_result:
                    curr = training_result['curriculum_progress']
                    print(f"   Curriculum Completion: {curr.get('completion_percentage', 0):.1f}%")
                    print(f"   Current Stage: {curr.get('current_stage', 'N/A')}")
            else:
                print(f"   Error: {training_result['error']}")
            
            print("\n✅ Full training example completed!")
            
        except Exception as e:
            logger.error(f"Full training example failed: {e}")
            print(f"❌ Full training example failed: {e}")
    
    def print_usage_guide(self):
        """Print usage guide and examples."""
        print("\n" + "="*80)
        print("📚 USAGE GUIDE")
        print("="*80)
        
        print("\n🔧 Installation:")
        print("   pip install -r requirements.txt")
        
        print("\n🚀 Quick Start:")
        print("   from tax_aware_main import TaxAwarePortfolioSystem")
        print("   system = TaxAwarePortfolioSystem()")
        print("   system.run_quick_demo()")
        
        print("\n🏋️ Training a Model:")
        print("   trainer = system.create_demo_trainer(['AAPL', 'MSFT', 'GOOGL'])")
        print("   result = trainer.train(num_episodes=1000)")
        
        print("\n🌐 Starting API Server:")
        print("   system.start_api_server(port=8000)")
        print("   # Access docs at: http://localhost:8000/docs")
        
        print("\n📊 Portfolio Analysis:")
        print("   from tax_aware_core import MarketDataManager")
        print("   data_manager = MarketDataManager(['AAPL', 'MSFT'])")
        print("   data = data_manager.fetch_data('2023-01-01', '2023-12-31')")
        
        print("\n💰 Tax Impact Calculation:")
        print("   from tax_aware_core import TaxCalculator")
        print("   tax_calc = TaxCalculator()")
        print("   impact = tax_calc.calculate_tax_impact(trades, tax_lots, date)")
        
        print("\n🎯 Available Methods:")
        methods = [
            "system.print_system_status() - Show system status",
            "system.run_quick_demo() - Run demonstration", 
            "system.run_full_training_example() - Full training demo",
            "system.create_demo_trainer() - Create trainer instance",
            "system.start_api_server() - Start REST API",
            "system.print_usage_guide() - Show this guide"
        ]
        
        for method in methods:
            print(f"   • {method}")
        
        print("\n📖 Documentation:")
        print("   • README: tax-aware-rl-system.md")
        print("   • API Docs: http://localhost:8000/docs (when server running)")
        print("   • Code Examples: See individual module docstrings")
        
        print("="*80)


def main():
    """Main application entry point."""
    print("\n🚀 Tax-Aware Portfolio Management RL System")
    print("Starting system initialization...")
    
    # Initialize system
    system = TaxAwarePortfolioSystem()
    
    # Show system status
    system.print_system_status()
    
    # Show usage guide
    system.print_usage_guide()
    
    # Prompt for demo
    print("\n" + "="*50)
    print("Choose an option:")
    print("1. Run Quick Demo")
    print("2. Run Full Training Example") 
    print("3. Start API Server")
    print("4. Exit")
    print("="*50)
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            system.run_quick_demo()
        elif choice == '2':
            system.run_full_training_example()
        elif choice == '3':
            port = input("Enter port (default 8000): ").strip() or "8000"
            try:
                port = int(port)
                system.start_api_server(port=port)
            except ValueError:
                print("Invalid port number")
        elif choice == '4':
            print("👋 Goodbye!")
        else:
            print("Invalid choice. Running quick demo instead...")
            system.run_quick_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")


if __name__ == "__main__":
    main()


print("✅ Main application module loaded successfully!")
print("   - Complete system interface")
print("   - Interactive demo capabilities") 
print("   - Training and API management")
print("   - Usage guide and documentation")
