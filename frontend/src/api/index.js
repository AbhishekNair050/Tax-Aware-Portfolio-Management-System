import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 minutes for long-running operations like training
  headers: {
    'Content-Type': 'application/json',
  },
});

// System API
export const getSystemStatus = () => api.get('/system/status');
export const getSystemInfo = () => api.get('/system/info');
export const getSystemHealth = () => api.get('/system/health');

// Training API
export const startTraining = (config) => api.post('/training/start', config, { timeout: 60000 }); // 1 minute for start
export const getTrainingStatus = (sessionId) => api.get(`/training/status/${sessionId}`);
export const stopTraining = (sessionId) => api.post(`/training/stop/${sessionId}`);
export const listTrainingSessions = () => api.get('/training/sessions');
export const getTrainingMetrics = (sessionId) => api.get(`/training/metrics/${sessionId}`);

// Portfolio API
export const getPortfolioState = () => api.get('/portfolio/state');
export const getPortfolioHistory = () => api.get('/portfolio/history');
export const getPortfolioPerformance = () => api.get('/portfolio/performance');
export const executeAction = (action) => api.post('/portfolio/action', action);

// Model API
export const listModels = () => api.get('/models/list');
export const loadModel = (modelPath) => api.post('/models/load', { model_path: modelPath });
export const saveModel = (modelPath) => api.post('/models/save', { model_path: modelPath });
export const getModelInfo = () => api.get('/models/info');

// Market Data API
export const getMarketData = (symbol, startDate, endDate) => 
  api.get(`/market/data/${symbol}`, { params: { start_date: startDate, end_date: endDate } });
export const getAvailableSymbols = () => api.get('/market/symbols');

// Tax Analysis API
export const getTaxAnalysis = () => api.get('/tax/analysis');
export const calculateTaxImpact = (trades) => api.post('/tax/calculate', { trades });

export default api;
