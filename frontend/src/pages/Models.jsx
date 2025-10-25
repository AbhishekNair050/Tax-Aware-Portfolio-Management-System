import { useState, useEffect } from 'react';
import { Upload, Download, Trash2, CheckCircle, RefreshCw } from 'lucide-react';
import Card from '../components/Card';
import Button from '../components/Button';
import { listModels, loadModel, getModelInfo } from '../api';

export default function Models() {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchModels = async () => {
    try {
      const response = await listModels();
      setModels(response.data.models || []);
      
      const infoResponse = await getModelInfo();
      setCurrentModel(infoResponse.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleLoadModel = async (modelPath) => {
    setLoading(true);
    try {
      await loadModel(modelPath);
      await fetchModels();
    } catch (error) {
      console.error('Error loading model:', error);
    } finally {
      setLoading(false);
    }
  };

  // Mock model data
  const modelsList = [
    {
      name: 'best_model.pth',
      path: '../models/best_model.pth',
      size: '45.2 MB',
      created: '2025-10-20',
      episodes: 1000,
      avgReward: 152.3,
      accuracy: null,
      isActive: true
    },
    {
      name: 'checkpoint_episode_25.pth',
      path: '../models/checkpoint_episode_25.pth',
      size: '45.1 MB',
      created: '2025-10-19',
      episodes: 25,
      avgReward: 138.7,
      accuracy: 92.1,
      isActive: false
    },
  ];

  // Use backend models if available, otherwise fallback to mock list
  const displayModels = (models && models.length) ? models : modelsList;

  const activeMeta = currentModel?.metadata || {};

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Models</h1>
          <p className="text-gray-600 mt-1">Manage trained RL models</p>
        </div>
        <div className="flex space-x-3">
          <Button onClick={fetchModels} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button variant="primary">
            <Upload className="w-4 h-4 mr-2" />
            Upload Model
          </Button>
        </div>
      </div>

      {/* Current Model Info */}
      <Card title="Active Model">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {(() => {
            const meta = currentModel?.metadata || currentModel?.model_info || {};
            const fallback = displayModels && displayModels.length ? displayModels[0] : {};
            const name = meta.name || meta.model_name || fallback.name || 'N/A';
            const episodes = meta.episodes || meta.training_episodes || fallback.episodes || '—';
            const avgReward = (meta.avgReward || meta.avg_reward || fallback.avgReward) ?? '—';
            const accuracy = (meta.accuracy || fallback.accuracy);

            return (
              <>
                <div>
                  <p className="text-sm text-gray-600">Model Name</p>
                  <p className="text-lg font-semibold text-gray-900 mt-1">{name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Training Episodes</p>
                  <p className="text-lg font-semibold text-gray-900 mt-1">{episodes}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Average Reward</p>
                  <p className="text-lg font-semibold text-green-600 mt-1">{typeof avgReward === 'number' ? `+${avgReward}` : avgReward}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Accuracy</p>
                  <p className="text-lg font-semibold text-primary-600 mt-1">{accuracy !== undefined && accuracy !== null ? `${accuracy}%` : '—'}</p>
                </div>
              </>
            );
          })()}
        </div>

        <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
          <div className="flex items-center">
            <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            <p className="text-sm text-green-800">
              This model is currently loaded and ready for inference
            </p>
          </div>
        </div>
      </Card>

      {/* Model Library */}
      <Card title="Model Library">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Name</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Size</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Created</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Episodes</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Avg Reward</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Accuracy</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Status</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayModels.map((model, idx) => (
                <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-4 px-4">
                    <div className="flex items-center">
                      <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center mr-3">
                        <span className="text-primary-600 font-bold text-sm">M</span>
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{model.name}</p>
                        <p className="text-xs text-gray-500">{model.path}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-4 px-4 text-sm text-gray-600">{model.size}</td>
                  <td className="py-4 px-4 text-sm text-gray-600">{model.created}</td>
                  <td className="py-4 px-4 text-sm text-gray-600">{model.episodes}</td>
                  <td className="py-4 px-4">
                    <span className="text-sm font-medium text-green-600">+{model.avgReward}</span>
                  </td>
                  <td className="py-4 px-4">
                    <span className="text-sm font-medium text-primary-600">{model.accuracy}%</span>
                  </td>
                  <td className="py-4 px-4">
                    {model.isActive ? (
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                        Active
                      </span>
                    ) : (
                      <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded-full">
                        Inactive
                      </span>
                    )}
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex space-x-2">
                      {!model.isActive && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleLoadModel(model.path)}
                          loading={loading}
                        >
                          Load
                        </Button>
                      )}
                      <Button variant="outline" size="sm">
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button variant="outline" size="sm">
                        <Trash2 className="w-4 h-4 text-red-600" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Model Performance Comparison */}
      <Card title="Model Performance Comparison">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Reward Comparison</h4>
            <div className="space-y-3">
              {modelsList.map((model, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">{model.name}</span>
                    <span className="font-medium text-gray-900">+{model.avgReward}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full" 
                      style={{ width: `${(model.avgReward / 160) * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Accuracy Comparison</h4>
            <div className="space-y-3">
              {modelsList.map((model, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">{model.name}</span>
                    <span className="font-medium text-gray-900">{model.accuracy}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary-500 h-2 rounded-full" 
                      style={{ width: `${model.accuracy}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Model Metadata */}
      <Card title="Model Training Details">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Architecture</p>
            <p className="text-sm text-gray-900">Soft Actor-Critic (SAC)</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">State Dim</p>
            <p className="text-sm text-gray-900">18</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Action Dim</p>
            <p className="text-sm text-gray-900">6</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Learning Rate</p>
            <p className="text-sm text-gray-900">0.0003</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Batch Size</p>
            <p className="text-sm text-gray-900">256</p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Replay Buffer</p>
            <p className="text-sm text-gray-900">1,000,000</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
