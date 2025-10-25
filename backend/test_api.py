"""
Test script to verify API endpoints are working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None):
    """Test an API endpoint."""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        status = "✅" if response.status_code == 200 else "❌"
        print(f"{status} {method} {path} - Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text[:100]}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {method} {path} - Error: {str(e)[:50]}")
        return False

def main():
    print("🧪 Testing Tax-Aware Portfolio API Endpoints\n")
    print("="*60)
    
    # Test System endpoints
    print("\n📊 System Endpoints:")
    test_endpoint("GET", "/")
    test_endpoint("GET", "/health")
    test_endpoint("GET", "/api/system/status")
    test_endpoint("GET", "/api/system/info")
    test_endpoint("GET", "/api/system/health")
    
    # Test Training endpoints
    print("\n🎓 Training Endpoints:")
    test_endpoint("GET", "/api/training/sessions")
    
    training_config = {
        "symbols": ["AAPL", "MSFT"],
        "initial_cash": 100000,
        "num_episodes": 10
    }
    test_endpoint("POST", "/api/training/start", training_config)
    
    # Test Portfolio endpoints
    print("\n💼 Portfolio Endpoints:")
    test_endpoint("GET", "/api/portfolio/state")
    test_endpoint("GET", "/api/portfolio/history")
    test_endpoint("GET", "/api/portfolio/performance")
    
    # Test Model endpoints
    print("\n🗄️ Model Endpoints:")
    test_endpoint("GET", "/api/models/list")
    test_endpoint("GET", "/api/models/info")
    
    # Test Market endpoints
    print("\n📈 Market Endpoints:")
    test_endpoint("GET", "/api/market/symbols")
    test_endpoint("GET", "/api/market/data/AAPL?start_date=2025-01-01&end_date=2025-10-20")
    
    # Test Tax endpoints
    print("\n💰 Tax Endpoints:")
    test_endpoint("GET", "/api/tax/analysis")
    
    print("\n" + "="*60)
    print("✨ API Testing Complete!\n")

if __name__ == "__main__":
    print("Make sure the API server is running: python run_server.py\n")
    import time
    time.sleep(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
