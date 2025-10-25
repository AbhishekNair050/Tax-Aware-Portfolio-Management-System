# Tax-Aware Portfolio Management - Frontend

Modern React-based frontend for the Tax-Aware Portfolio Management system.

## 🎨 Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization
- **Lucide React** - Icon library
- **React Router** - Navigation
- **Axios** - HTTP client

## 📦 Installation

```bash
npm install
```

## 🚀 Development

```bash
npm run dev
```

Opens at http://localhost:3000

## 🏗️ Build

```bash
npm run build
```

Outputs to `dist/` folder

## 📁 Project Structure

```
src/
├── components/      # Reusable UI components
│   ├── Layout.jsx      # Main layout with sidebar
│   ├── Card.jsx        # Card container
│   ├── Button.jsx      # Button component
│   └── StatCard.jsx    # Statistics card
├── pages/           # Page components
│   ├── Dashboard.jsx   # Main dashboard
│   ├── Training.jsx    # Training management
│   ├── Portfolio.jsx   # Portfolio view
│   ├── Analytics.jsx   # Analytics & reports
│   └── Models.jsx      # Model management
├── api/             # API client
│   └── index.js        # API endpoints
├── App.jsx          # Main app component
├── main.jsx         # Entry point
└── index.css        # Global styles
```

## 🎨 Features

### Dashboard
- Real-time portfolio overview
- Performance charts
- Asset allocation
- Recent activity
- System status

### Training
- Start/stop training sessions
- Configure training parameters
- Real-time metrics visualization
- Training history

### Portfolio
- Current holdings table
- Execute trades
- Asset allocation pie chart
- Tax impact analysis

### Analytics
- Performance vs benchmark
- Risk-adjusted metrics
- Monthly returns
- Tax savings analysis
- Key insights

### Models
- Browse available models
- Load/switch models
- Compare performance
- View model metadata

## 🔧 Configuration

### API Proxy

Edit `vite.config.js`:

```js
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

### Tailwind Theme

Edit `tailwind.config.js` to customize colors and styling.

## 📊 API Integration

All API calls are centralized in `src/api/index.js`:

```javascript
import { getSystemStatus, startTraining, getPortfolioState } from './api';

// Use in components
const data = await getSystemStatus();
```

## 🎯 Adding New Pages

1. Create page component in `src/pages/`
2. Add route in `App.jsx`
3. Add navigation item in `Layout.jsx`

## 💅 Styling Guide

Uses Tailwind CSS utility classes:

```jsx
<div className="bg-white rounded-lg shadow-sm p-6">
  <h2 className="text-xl font-bold text-gray-900">Title</h2>
</div>
```

## 🔌 Environment Variables

Create `.env` for custom configuration:

```
VITE_API_URL=http://localhost:8000
```

## 📱 Responsive Design

All components are mobile-responsive with:
- Mobile-first approach
- Responsive grid layouts
- Collapsible sidebar
- Touch-friendly controls

## 🚀 Deployment

Build for production:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

Deploy the `dist/` folder to any static hosting service.

## 🎨 Component Library

### Button
```jsx
<Button variant="primary" size="md" onClick={handleClick}>
  Click Me
</Button>
```

Variants: `primary`, `secondary`, `success`, `danger`, `outline`

### Card
```jsx
<Card title="Card Title" action={<Button>Action</Button>}>
  Card content
</Card>
```

### StatCard
```jsx
<StatCard
  title="Total Value"
  value="$1,320,000"
  change={25.0}
  trend="up"
  icon={DollarSign}
/>
```

## 📈 Charts

Uses Recharts library:

```jsx
<ResponsiveContainer width="100%" height={300}>
  <LineChart data={data}>
    <Line dataKey="value" stroke="#0ea5e9" />
  </LineChart>
</ResponsiveContainer>
```

## 🐛 Debugging

- Open browser console (F12)
- Check Network tab for API calls
- Enable React DevTools extension

## 📝 License

MIT
