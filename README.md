# Tax-Aware Portfolio Management System

A sophisticated reinforcement learning system for tax-aware portfolio management with a beautiful web interface.

## 🎯 Features

- **🤖 Reinforcement Learning**: State-of-the-art Soft Actor-Critic (SAC) algorithm
- **💰 Tax Optimization**: Minimize tax liability through intelligent trading
- **📊 Real-time Analytics**: Comprehensive performance and risk metrics
- **🎨 Modern UI**: Beautiful React-based dashboard with interactive charts
- **📈 Market Integration**: Real-time market data via yfinance
- **🔄 Curriculum Learning**: Progressive training for better convergence

## 🏗️ Project Structure

```
RL/
├── backend/           # Python backend API
│   ├── tax_aware_api.py        # FastAPI REST API
│   ├── tax_aware_core.py       # Core portfolio logic
│   ├── tax_aware_rl.py         # RL environment & agent
│   ├── tax_aware_training.py   # Training manager
│   ├── tax_aware_main.py       # Main entry point
│   └── requirements.txt        # Python dependencies
├── frontend/          # React frontend
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Dashboard, Training, Portfolio, Analytics, Models
│   │   └── api/          # API client
│   ├── package.json
│   └── vite.config.js
├── models/            # Trained RL models
├── docs/              # Documentation
└── start.ps1          # Startup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- 4GB RAM minimum (8GB recommended)

### Installation & Running

Simply run the startup script:

```powershell
.\start.ps1
```

This will:
1. Check for Python and Node.js
2. Install all dependencies
3. Start the backend API server (http://localhost:8000)
4. Start the frontend dev server (http://localhost:3000)

### Manual Installation

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python tax_aware_api.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📱 Using the Application

### Dashboard
- Real-time portfolio overview
- Performance charts vs benchmark
- Asset allocation visualization
- System status monitoring

### Training
- Configure training parameters
- Monitor training progress
- View training metrics in real-time
- Manage training sessions

### Portfolio
- View current holdings
- Execute trades
- Analyze tax impact
- Track gains/losses

### Analytics
- Performance vs benchmark
- Risk-adjusted metrics
- Tax savings analysis
- Monthly returns breakdown

### Models
- Browse trained models
- Load/unload models
- Compare model performance
- View model metadata

## 🔧 Configuration

### Backend Configuration

Edit `backend/tax_aware_api.py` to configure:
- Port (default: 8000)
- CORS settings
- Model paths
- Training parameters

### Frontend Configuration

Edit `frontend/vite.config.js` to configure:
- Port (default: 3000)
- API proxy settings

## 📊 API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `GET /api/system/status` - System status
- `POST /api/training/start` - Start training
- `GET /api/portfolio/state` - Get portfolio state
- `POST /api/portfolio/action` - Execute trade
- `GET /api/models/list` - List available models

## 🎓 Training a New Model

1. Navigate to the **Training** page
2. Configure training parameters:
   - Trading symbols (e.g., AAPL, MSFT, GOOGL)
   - Initial cash
   - Number of episodes
   - Training date range
3. Click **Start Training**
4. Monitor progress in real-time
5. Model is automatically saved in `models/`

## 🧪 Technology Stack

### Backend
- **Framework**: FastAPI
- **RL Library**: Stable-Baselines3, PyTorch
- **Data**: yfinance, pandas, numpy
- **Analysis**: ta (technical analysis)

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Routing**: React Router
- **HTTP**: Axios

## 📈 Performance

- **Training Speed**: ~100-200 episodes/minute (depending on hardware)
- **Inference**: Real-time (<10ms per decision)
- **Memory Usage**: ~2-4GB during training
- **Model Size**: ~45MB per checkpoint

## 🔐 Security

- CORS enabled for localhost development
- API authentication ready (currently disabled for development)
- Input validation on all endpoints
- Secure model loading

## 🤝 Contributing

This is a personal project. Feel free to fork and adapt for your needs.

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Stable-Baselines3 for RL implementations
- yfinance for market data
- React and Vite communities

## 📞 Support

For issues or questions, please check the documentation in the `docs/` folder.

---

**Made with ❤️ using React, FastAPI, and Reinforcement Learning**
