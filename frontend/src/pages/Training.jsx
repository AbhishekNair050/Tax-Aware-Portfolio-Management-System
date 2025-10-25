import { useState, useEffect } from 'react';
import { Play, Square, Download, RefreshCw } from 'lucide-react';
import Card from '../components/Card';
import Button from '../components/Button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { startTraining, getTrainingStatus, stopTraining, listTrainingSessions } from '../api';

export default function Training() {
  const [isTraining, setIsTraining] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [config, setConfig] = useState({
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    initial_cash: 1000000,
    num_episodes: 1000,
    train_start: '2020-01-01',
    train_end: '2022-12-31',
    val_start: '2023-01-01',
    val_end: '2023-12-31',
    use_curriculum: true,
  });
  const [currentSession, setCurrentSession] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState([
    { episode: 0, reward: -50, loss: 2.5, value: 950000 },
    { episode: 100, reward: 20, loss: 1.8, value: 1020000 },
    { episode: 200, reward: 45, loss: 1.2, value: 1100000 },
    { episode: 300, reward: 80, loss: 0.8, value: 1180000 },
    { episode: 400, reward: 120, loss: 0.5, value: 1280000 },
    { episode: 500, reward: 150, loss: 0.3, value: 1350000 },
  ]);

  const handleStartTraining = async () => {
    try {
      setIsTraining(true);
      console.log('Starting training with config:', config);
      const response = await startTraining(config);
      console.log('Training started:', response.data);
      setCurrentSession(response.data);
      alert(`✅ Training started successfully! Session ID: ${response.data.session_id}`);
    } catch (error) {
      console.error('Error starting training:', error);
      alert(`❌ Error starting training: ${error.message}`);
      setIsTraining(false);
    }
  };

  const handleStopTraining = async () => {
    if (currentSession) {
      try {
        console.log('Stopping training:', currentSession.session_id);
        await stopTraining(currentSession.session_id);
        setIsTraining(false);
        alert('⏹️ Training stopped');
      } catch (error) {
        console.error('Error stopping training:', error);
        alert(`❌ Error stopping training: ${error.message}`);
      }
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await listTrainingSessions();
      console.log('Sessions fetched:', response.data);
      setSessions(response.data.sessions || []);
    } catch (error) {
      console.error('Error fetching sessions:', error);
    }
  };

  const pollTrainingStatus = async () => {
    if (currentSession && isTraining) {
      try {
        const response = await getTrainingStatus(currentSession.session_id);
        console.log('Training status:', response.data);
        setTrainingProgress(response.data);
        
        if (response.data.status === 'completed' || response.data.status === 'stopped') {
          setIsTraining(false);
          alert(`✅ Training ${response.data.status}!`);
        }
      } catch (error) {
        console.error('Error polling training status:', error);
      }
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  // Poll for training updates every 2 seconds
  useEffect(() => {
    if (isTraining && currentSession) {
      const interval = setInterval(pollTrainingStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [isTraining, currentSession]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Training</h1>
        <p className="text-gray-600 mt-1">Train and manage RL models</p>
      </div>

      {/* Training Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Training Configuration">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Trading Symbols
                </label>
                <input
                  type="text"
                  value={config.symbols.join(', ')}
                  onChange={(e) => setConfig({ ...config, symbols: e.target.value.split(',').map(s => s.trim()) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="AAPL, MSFT, GOOGL"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Initial Cash
                </label>
                <input
                  type="number"
                  value={config.initial_cash}
                  onChange={(e) => setConfig({ ...config, initial_cash: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Episodes
                </label>
                <input
                  type="number"
                  value={config.num_episodes}
                  onChange={(e) => setConfig({ ...config, num_episodes: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Train Period
                </label>
                <div className="flex space-x-2">
                  <input
                    type="date"
                    value={config.train_start}
                    onChange={(e) => setConfig({ ...config, train_start: e.target.value })}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                  <input
                    type="date"
                    value={config.train_end}
                    onChange={(e) => setConfig({ ...config, train_end: e.target.value })}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="col-span-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.use_curriculum}
                    onChange={(e) => setConfig({ ...config, use_curriculum: e.target.checked })}
                    className="w-4 h-4 text-primary-500 border-gray-300 rounded focus:ring-primary-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Enable Curriculum Learning</span>
                </label>
              </div>
            </div>

            <div className="mt-6 flex space-x-3">
              {!isTraining ? (
                <Button onClick={handleStartTraining} variant="success">
                  <Play className="w-4 h-4 mr-2" />
                  Start Training
                </Button>
              ) : (
                <Button onClick={handleStopTraining} variant="danger">
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </Button>
              )}
            </div>
          </Card>
        </div>

        {/* Training Status */}
        <Card title="Current Status">
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-600">Status</p>
              <p className="text-lg font-semibold text-gray-900">
                {trainingProgress?.status || (isTraining ? 'Training' : 'Idle')}
              </p>
            </div>
            
            {currentSession && (
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-xs text-blue-600 font-medium">Session ID</p>
                <p className="text-xs text-blue-800 font-mono truncate">
                  {currentSession.session_id}
                </p>
              </div>
            )}
            
            {isTraining && trainingProgress && (
              <>
                <div>
                  <p className="text-sm text-gray-600">Current Episode</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {trainingProgress.current_episode || 0} / {config.num_episodes}
                  </p>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary-500 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${trainingProgress.progress || 0}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {(trainingProgress.progress || 0).toFixed(1)}% complete
                  </p>
                </div>

                {trainingProgress.metrics?.avg_reward !== undefined && (
                  <div>
                    <p className="text-sm text-gray-600">Average Reward</p>
                    <p className="text-lg font-semibold text-green-600">
                      {trainingProgress.metrics.avg_reward >= 0 ? '+' : ''}
                      {trainingProgress.metrics.avg_reward.toFixed(1)}
                    </p>
                  </div>
                )}

                {trainingProgress.metrics?.loss !== undefined && (
                  <div>
                    <p className="text-sm text-gray-600">Loss</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {trainingProgress.metrics.loss.toFixed(3)}
                    </p>
                  </div>
                )}
              </>
            )}
            
            {!isTraining && !currentSession && (
              <div className="text-center py-6 text-gray-500">
                <p className="text-sm">No active training session</p>
                <p className="text-xs mt-1">Configure and start training to begin</p>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Training Metrics Chart */}
      <Card title="Training Metrics">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={trainingMetrics}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="episode" stroke="#6b7280" />
            <YAxis yAxisId="left" stroke="#6b7280" />
            <YAxis yAxisId="right" orientation="right" stroke="#6b7280" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px'
              }}
            />
            <Legend />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="reward" 
              stroke="#10b981" 
              strokeWidth={2}
              name="Reward"
            />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="loss" 
              stroke="#ef4444" 
              strokeWidth={2}
              name="Loss"
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="value" 
              stroke="#0ea5e9" 
              strokeWidth={2}
              name="Portfolio Value"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Training History */}
      <Card 
        title="Training History"
        action={
          <Button variant="outline" size="sm" onClick={fetchSessions}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Session ID</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Date</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Episodes</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Status</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Final Reward</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {[
                { id: 'session-001', date: '2025-10-20', episodes: 1000, status: 'Completed', reward: 152.3 },
                { id: 'session-002', date: '2025-10-19', episodes: 800, status: 'Completed', reward: 138.7 },
                { id: 'session-003', date: '2025-10-18', episodes: 500, status: 'Stopped', reward: 95.4 },
              ].map((session, idx) => (
                <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-sm text-gray-900">{session.id}</td>
                  <td className="py-3 px-4 text-sm text-gray-600">{session.date}</td>
                  <td className="py-3 px-4 text-sm text-gray-600">{session.episodes}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      session.status === 'Completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {session.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm font-medium text-green-600">+{session.reward}</td>
                  <td className="py-3 px-4">
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
