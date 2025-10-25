import { useState, useEffect, useMemo } from 'react';
import Card from '../components/Card';
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getPortfolioHistory, getPortfolioPerformance, getTaxAnalysis } from '../api';

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(2)}%`;
};

const formatRatio = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return Number(value).toFixed(2);
};

export default function Analytics() {
  const [timeRange, setTimeRange] = useState('3M');
  const [history, setHistory] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [tax, setTax] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [historyRes, perfRes, taxRes] = await Promise.all([
          getPortfolioHistory(),
          getPortfolioPerformance(),
          getTaxAnalysis()
        ]);

        setHistory(historyRes.data?.history || []);
        setPerformance(perfRes.data || null);
        setTax(taxRes.data || null);
        setError(null);
      } catch (err) {
        console.error('Failed to load analytics data', err);
        setError('Unable to load analytics data. Please verify the backend service.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  const normalizedHistory = useMemo(() => {
    return history
      .map((point) => {
        const date = point.date ? new Date(point.date) : null;
        if (!date || Number.isNaN(Number(point.value))) {
          return null;
        }
        return {
          date,
          label: date.toLocaleDateString(),
          value: Number(point.value),
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.date - b.date);
  }, [history]);

  const filteredHistory = useMemo(() => {
    if (!normalizedHistory.length) {
      return [];
    }

    const cutoff = (() => {
      const last = normalizedHistory[normalizedHistory.length - 1].date;
      switch (timeRange) {
        case '1W':
          return new Date(last.getTime() - 7 * 24 * 60 * 60 * 1000);
        case '1M':
          return new Date(last.getTime() - 30 * 24 * 60 * 60 * 1000);
        case '3M':
          return new Date(last.getTime() - 90 * 24 * 60 * 60 * 1000);
        case '6M':
          return new Date(last.getTime() - 180 * 24 * 60 * 60 * 1000);
        case '1Y':
          return new Date(last.getTime() - 365 * 24 * 60 * 60 * 1000);
        default:
          return null;
      }
    })();

    if (!cutoff) {
      return normalizedHistory;
    }

    return normalizedHistory.filter((point) => point.date >= cutoff);
  }, [normalizedHistory, timeRange]);

  const performanceSeries = filteredHistory.map((point) => ({
    date: point.label,
    value: point.value,
  }));

  const monthlyReturns = useMemo(() => {
    const monthBuckets = new Map();
    filteredHistory.forEach((point) => {
      const key = `${point.date.getFullYear()}-${point.date.getMonth() + 1}`;
      const entry = monthBuckets.get(key) || {
        label: point.date.toLocaleString('default', { month: 'short', year: 'numeric' }),
        first: point.value,
        last: point.value,
      };
      entry.last = point.value;
      monthBuckets.set(key, entry);
    });

    return Array.from(monthBuckets.values()).map((entry) => ({
      month: entry.label,
      returns: entry.first ? ((entry.last - entry.first) / entry.first) * 100 : 0,
    }));
  }, [filteredHistory]);

  const riskMetrics = [
    { metric: 'Total Return', value: performance?.total_return, type: 'percent' },
    { metric: 'Sharpe Ratio', value: performance?.sharpe_ratio, type: 'ratio' },
    { metric: 'Max Drawdown', value: performance?.max_drawdown, type: 'percent' },
    { metric: 'Volatility', value: performance?.volatility, type: 'percent' },
    { metric: 'Win Rate', value: performance?.win_rate, type: 'percent' },
    { metric: 'Tax Efficiency', value: performance?.tax_efficiency, type: 'percent' },
  ];

  const cumulativeTaxSeries = useMemo(() => {
    if (!tax?.unrealized_positions?.length) {
      return [];
    }
    let cumulative = 0;
    return tax.unrealized_positions
      .slice()
      .sort((a, b) => {
        const aDate = new Date(a.purchase_date || a.timestamp || Date.now());
        const bDate = new Date(b.purchase_date || b.timestamp || Date.now());
        return aDate - bDate;
      })
      .map((lot) => {
        const gain = Number(lot.unrealized_gain || 0);
        cumulative += gain;
        const labelDate = lot.purchase_date || lot.timestamp;
        const formatted = labelDate ? new Date(labelDate).toLocaleDateString() : 'Latest';
        return {
          date: formatted,
          value: cumulative,
        };
      });
  }, [tax]);

  const taxBreakdown = tax
    ? [
        { label: 'Short-term Realized', value: tax.short_term_realized },
        { label: 'Long-term Realized', value: tax.long_term_realized },
        { label: 'Unrealized Gains', value: tax.unrealized_gains },
        { label: 'Tax Liability', value: tax.tax_liability },
      ]
    : [];

  const insights = useMemo(() => {
    const items = [];

    if (performance?.total_return !== undefined) {
      items.push({
        title: 'Performance',
        text: `Total return sits at ${formatPercent(performance.total_return)} over the selected interval.`,
        tone: performance.total_return >= 0 ? 'positive' : 'neutral',
      });
    }

    if (tax?.tax_liability !== undefined) {
      items.push({
        title: 'Tax Exposure',
        text: `Projected tax liability is $${Number(tax.tax_liability || 0).toLocaleString()}.`,
        tone: 'informational',
      });
    }

    if (tax?.tax_efficiency_score !== undefined) {
      items.push({
        title: 'Tax Efficiency',
        text: `Tax efficiency score is ${formatPercent(tax.tax_efficiency_score)}.`,
        tone: 'positive',
      });
    }

    if (tax?.recommendations?.length) {
      tax.recommendations.slice(0, 2).forEach((recommendation, index) => {
        items.push({
          title: `Recommendation ${index + 1}`,
          text: recommendation,
          tone: 'neutral',
        });
      });
    }

    return items;
  }, [performance, tax]);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600 mt-1">Detailed performance and risk analysis</p>
        </div>
        <div className="flex space-x-2">
          {['1W', '1M', '3M', '6M', '1Y', 'ALL'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                timeRange === range
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      <Card title="Portfolio Value">
        {performanceSeries.length > 1 ? (
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={performanceSeries}>
              <defs>
                <linearGradient id="valueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" tickFormatter={(value) => `$${Number(value).toLocaleString()}`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
                formatter={(value) => `$${Number(value).toLocaleString()}`}
              />
              <Area type="monotone" dataKey="value" stroke="#0ea5e9" fill="url(#valueGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="p-6 text-sm text-gray-500">Not enough data to render the value chart.</div>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Risk Metrics">
          <div className="space-y-4">
            {riskMetrics.map((metric) => {
              const formattedValue = metric.type === 'ratio'
                ? formatRatio(metric.value)
                : formatPercent(metric.value);

              const progressWidth = (() => {
                if (metric.type !== 'percent') {
                  return null;
                }
                const numeric = Number(metric.value);
                if (Number.isNaN(numeric)) {
                  return '0%';
                }
                return `${Math.min(Math.abs(numeric), 100)}%`;
              })();

              return (
                <div key={metric.metric} className="border-b border-gray-100 pb-3 last:border-0">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">{metric.metric}</span>
                    <span className="text-sm font-semibold text-gray-900">{formattedValue}</span>
                  </div>
                  {progressWidth && (
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full"
                        style={{ width: progressWidth }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>

        <Card title="Monthly Returns">
          {monthlyReturns.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={monthlyReturns}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" stroke="#6b7280" />
                <YAxis
                  stroke="#6b7280"
                  tickFormatter={(value) => {
                    if (typeof value !== 'number' || Number.isNaN(value)) {
                      return '--';
                    }
                    return `${value.toFixed(1)}%`;
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                  formatter={(value) => {
                    const numeric = Number(value);
                    if (Number.isNaN(numeric)) {
                      return '--';
                    }
                    return `${numeric.toFixed(2)}%`;
                  }}
                />
                <Bar dataKey="returns" fill="#10b981" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="p-6 text-sm text-gray-500">Monthly return breakdown requires additional history.</div>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Cumulative Tax Impact">
          {cumulativeTaxSeries.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={cumulativeTaxSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" tickFormatter={(value) => `$${Number(value).toLocaleString()}`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                  formatter={(value) => `$${Number(value).toLocaleString()}`}
                />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="p-6 text-sm text-gray-500">No unrealized tax impact data available.</div>
          )}
        </Card>

        <Card title="Tax Summary">
          <div className="space-y-3">
            {taxBreakdown.map((item) => (
              <div key={item.label} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium text-gray-900">{item.label}</p>
                </div>
                <p className="text-sm font-semibold text-gray-700">$ {Number(item.value || 0).toLocaleString()}</p>
              </div>
            ))}
            <div className="flex items-center justify-between p-3 bg-primary-50 rounded-lg border-2 border-primary-200">
              <p className="text-sm font-bold text-gray-900">Tax Efficiency Score</p>
              <p className="text-lg font-bold text-primary-600">{formatPercent(tax?.tax_efficiency_score)}</p>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Key Insights">
        {loading ? (
          <div className="p-6 text-sm text-gray-500">Loading insights…</div>
        ) : insights.length ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {insights.map((insight) => {
              const toneClasses = {
                positive: 'bg-green-50 border-green-200 text-green-800',
                informational: 'bg-blue-50 border-blue-200 text-blue-800',
                neutral: 'bg-gray-50 border-gray-200 text-gray-700',
              };

              return (
                <div
                  key={insight.title}
                  className={`p-4 rounded-lg border ${toneClasses[insight.tone] || toneClasses.neutral}`}
                >
                  <h4 className="text-sm font-semibold mb-2">{insight.title}</h4>
                  <p className="text-xs leading-relaxed">{insight.text}</p>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-6 text-sm text-gray-500">No insights available yet.</div>
        )}
      </Card>
    </div>
  );
}
