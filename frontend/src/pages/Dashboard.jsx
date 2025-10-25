import { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, Activity, Percent, RefreshCw, AlertTriangle } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import Button from '../components/Button';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getSystemStatus, getPortfolioState, getPortfolioPerformance, getPortfolioHistory } from '../api';

const formatPercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '--';
  }
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(1)}%`;
};

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [statusRes, portfolioRes, perfRes, historyRes] = await Promise.all([
        getSystemStatus(),
        getPortfolioState(),
        getPortfolioPerformance(),
        getPortfolioHistory()
      ]);
      
      setSystemStatus(statusRes.data);
      setPortfolio(portfolioRes.data);
      setPerformance(perfRes.data);
      setHistory(historyRes.data?.history || []);
      setError(null);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError('Unable to load dashboard data. Please check the backend logs.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const portfolioHistory = history
    .map((point) => ({
      date: point.date ? new Date(point.date).toLocaleDateString() : '',
      value: point.value,
    }))
    .filter((point) => point.value !== undefined);

  const assetAllocation = (portfolio?.positions || []).map((pos) => ({
    name: pos.symbol,
    value: pos.value,
  }));

  if (portfolio?.cash) {
    assetAllocation.push({ name: 'Cash', value: portfolio.cash });
  }

  const recentTrades = portfolio?.recent_trades || [];

  const systemStatusItems = [
    {
      component: 'API Server',
      status: systemStatus?.status ? systemStatus.status.toUpperCase() : 'UNKNOWN',
      statusClass: systemStatus?.status === 'operational' ? 'bg-green-500' : 'bg-yellow-500',
    },
    {
      component: 'Active Trainers',
      status: systemStatus?.active_trainers ?? 0,
      statusClass: (systemStatus?.active_trainers ?? 0) > 0 ? 'bg-green-500' : 'bg-gray-400',
    },
    {
      component: 'Trained Models',
      status: systemStatus?.trained_models ?? 0,
      statusClass: (systemStatus?.trained_models ?? 0) > 0 ? 'bg-green-500' : 'bg-gray-400',
    },
    {
      component: 'Last Update',
      status: systemStatus?.timestamp ? new Date(systemStatus.timestamp).toLocaleString() : '--',
      statusClass: 'bg-blue-500',
    },
  ];

  if (loading && !portfolio) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Real-time portfolio and system overview</p>
        </div>
        <Button onClick={fetchData} variant="outline">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="flex items-center space-x-3 p-4 rounded-lg bg-red-50 border border-red-200 text-red-700">
          <AlertTriangle className="w-5 h-5" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Portfolio Value"
          value={portfolio?.total_value !== undefined ? `$${portfolio.total_value.toLocaleString()}` : '--'}
          change={Number.isFinite(portfolio?.gain_percent) ? portfolio.gain_percent : undefined}
          trend="up"
          icon={DollarSign}
        />
        <StatCard
          title="Total Returns"
          value={formatPercent(portfolio?.gain_percent)}
          trend="up"
          icon={TrendingUp}
        />
        <StatCard
          title="Tax Efficiency"
          value={formatPercent(performance?.tax_efficiency)}
          trend="up"
          icon={Percent}
        />
        <StatCard
          title="Active Positions"
          value={portfolio?.positions?.length ?? 0}
          icon={Activity}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Performance Chart */}
        <Card title="Portfolio Performance">
          {portfolioHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={portfolioHistory}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#0ea5e9" 
                  fillOpacity={1} 
                  fill="url(#colorValue)"
                  name="Portfolio"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="p-6 text-sm text-gray-500">No portfolio history available yet.</div>
          )}
        </Card>

        {/* Asset Allocation Chart */}
        <Card title="Asset Allocation">
          {assetAllocation.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={assetAllocation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => `$${Number(value).toLocaleString()}`}
                />
                <Bar dataKey="value" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="p-6 text-sm text-gray-500">No open positions to visualize.</div>
          )}
        </Card>
      </div>

      {/* Recent Activity & System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Recent Activity">
          {recentTrades.length > 0 ? (
            <div className="space-y-4">
              {recentTrades.slice().reverse().map((trade, idx) => {
                const dateLabel = trade.date ? new Date(trade.date).toLocaleString() : '';
                const isBuy = trade.quantity > 0;
                return (
                  <div key={`${trade.trade_id}-${idx}`} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
                    <div>
                      <p className="font-medium text-gray-900">
                        <span className={isBuy ? 'text-green-600' : 'text-red-600'}>
                          {isBuy ? 'Buy' : 'Sell'}
                        </span>
                        {' '}{Math.abs(trade.quantity)} shares of {trade.symbol}
                      </p>
                      <p className="text-sm text-gray-500">
                        ${trade.price?.toFixed(2)} • {dateLabel}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="p-4 text-sm text-gray-500">No recent trades recorded.</div>
          )}
        </Card>

        <Card title="System Status">
          <div className="space-y-4">
            {systemStatusItems.map((item) => (
              <div key={item.component} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
                <span className="text-gray-700 font-medium">{item.component}</span>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full ${item.statusClass} mr-2`}></div>
                  <span className="text-sm text-gray-600">{item.status}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
