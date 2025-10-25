import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, DollarSign, RefreshCw } from 'lucide-react';
import Card from '../components/Card';
import Button from '../components/Button';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { getPortfolioState, getPortfolioHistory, executeAction, getTaxAnalysis } from '../api';

const formatPercent = (value) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '--';
  return `${num >= 0 ? '+' : ''}${num.toFixed(1)}%`;
};

const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [actionType, setActionType] = useState('buy');
  const [shares, setShares] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchPortfolio = async () => {
    try {
      const response = await getPortfolioState();
      console.log('Portfolio data:', response.data);
      setPortfolio(response.data);
      // tax analysis will be loaded separately
    } catch (error) {
      console.error('Error fetching portfolio:', error);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  // Local tax analysis state
  const [taxAnalysis, setTaxAnalysis] = useState(null);

  useEffect(() => {
    const loadTax = async () => {
      try {
        const res = await getTaxAnalysis();
        setTaxAnalysis(res.data || null);
      } catch (err) {
        console.warn('Tax analysis fetch failed', err);
        setTaxAnalysis(null);
      }
    };
    loadTax();
  }, [portfolio]);

  const handleExecuteAction = async () => {
    if (!selectedSymbol || !shares || shares <= 0) {
      alert('⚠️ Please enter a valid symbol and number of shares');
      return;
    }
    
    setLoading(true);
    try {
      console.log('Executing action:', { symbol: selectedSymbol, action: actionType, shares });
      const response = await executeAction({
        symbol: selectedSymbol,
        action: actionType,
        shares: shares
      });
      console.log('Action executed:', response.data);
      alert(`✅ ${actionType === 'buy' ? 'Bought' : 'Sold'} ${shares} shares of ${selectedSymbol}`);
      await fetchPortfolio();
      setShares('');
      setSelectedSymbol('');
    } catch (error) {
      console.error('Error executing action:', error);
      alert(`❌ Error executing trade: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Use API data or fallback to mock data
  const holdings = portfolio?.positions || [
    { symbol: 'AAPL', shares: 500, avgPrice: 170.50, currentPrice: 175.50, value: 87750, gain: 2500 },
    { symbol: 'MSFT', shares: 300, avgPrice: 375.00, currentPrice: 380.25, value: 114075, gain: 1575 },
    { symbol: 'GOOGL', shares: 400, avgPrice: 138.20, currentPrice: 140.80, value: 56320, gain: 1040 },
  ];

  const pieData = holdings.map(h => ({ name: h.symbol, value: h.value || 0 }));

  const totalValue = portfolio?.total_value || holdings.reduce((sum, h) => sum + (h.value || 0), 0);
  const totalGain = portfolio?.total_gain || holdings.reduce((sum, h) => sum + (h.gain || 0), 0);
  const gainPercent = portfolio?.gain_percent || (totalValue > totalGain ? ((totalGain / (totalValue - totalGain)) * 100).toFixed(2) : 0);

  console.log('Portfolio render:', { portfolio, holdings, totalValue, totalGain, gainPercent });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Portfolio</h1>
          <p className="text-gray-600 mt-1">Manage your portfolio positions</p>
        </div>
        <Button onClick={fetchPortfolio} variant="outline">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Value</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                ${totalValue.toLocaleString()}
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <DollarSign className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Gain</p>
              <p className="text-2xl font-bold text-green-600 mt-1">
                +${totalGain.toLocaleString()}
              </p>
              <p className="text-sm text-gray-500 mt-1">+{gainPercent}%</p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Cash Available</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                ${(portfolio?.cash || (1000000 - totalValue)).toLocaleString()}
              </p>
            </div>
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
              <DollarSign className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </Card>
      </div>

      {/* Holdings and Allocation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Holdings Table */}
        <div className="lg:col-span-2">
          <Card title="Holdings">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Symbol</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Shares</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Avg Price</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Current</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Value</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Gain/Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((holding, idx) => {
                    const avgPrice = Number(holding.avgPrice) || 0;
                    const currentPrice = Number(holding.currentPrice) || 0;
                    const value = Number(holding.value) || 0;
                    const gain = Number(holding.gain) || 0;
                    
                    return (
                      <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-3 px-4">
                          <span className="font-semibold text-gray-900">{holding.symbol}</span>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600">{holding.shares || 0}</td>
                        <td className="py-3 px-4 text-sm text-gray-600">${avgPrice.toFixed(2)}</td>
                        <td className="py-3 px-4 text-sm font-medium text-gray-900">
                          ${currentPrice.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-sm font-medium text-gray-900">
                          ${value.toLocaleString()}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center">
                            {gain >= 0 ? (
                              <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                            ) : (
                              <TrendingDown className="w-4 h-4 text-red-600 mr-1" />
                            )}
                            <span className={`text-sm font-medium ${gain >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              ${Math.abs(gain).toLocaleString()}
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Asset Allocation Pie Chart */}
        <Card title="Asset Allocation">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Trade Execution */}
      <Card title="Execute Trade">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Action
            </label>
            <select
              value={actionType}
              onChange={(e) => setActionType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Symbol
            </label>
            <input
              type="text"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="AAPL"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Shares
            </label>
            <input
              type="number"
              value={shares}
              onChange={(e) => setShares(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="100"
              min="1"
            />
          </div>

          <div className="flex items-end">
            <Button
              onClick={handleExecuteAction}
              loading={loading}
              disabled={!selectedSymbol || !shares || Number(shares) <= 0}
              className="w-full"
            >
              Execute Trade
            </Button>
          </div>
        </div>
      </Card>

      {/* Tax Impact */}
      <Card title="Tax Impact Analysis">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-600">Unrealized Gains</p>
            <p className="text-xl font-bold text-gray-900 mt-1">
              ${totalGain.toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">Tax-deferred</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Estimated Tax Liability</p>
            <p className="text-xl font-bold text-orange-600 mt-1">
              ${(totalGain * 0.20).toLocaleString()}
            </p>
            <p className="text-xs text-gray-500 mt-1">20% long-term rate</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Tax Efficiency Score</p>
            <p className="text-xl font-bold text-green-600 mt-1">
              {taxAnalysis?.tax_efficiency_score !== undefined && taxAnalysis?.tax_efficiency_score !== null
                ? `${Number(taxAnalysis.tax_efficiency_score).toFixed(1)}%`
                : '--'}
            </p>
            <p className="text-xs text-gray-500 mt-1">{(taxAnalysis?.tax_efficiency_score || 0) >= 75 ? 'Excellent' : 'Review'}</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
