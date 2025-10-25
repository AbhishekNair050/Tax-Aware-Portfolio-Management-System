"""
Quick start script for the backend API server
Run this to start the FastAPI server
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Tax-Aware Portfolio Management API Server...")
    print("📚 API Documentation will be available at http://localhost:8000/docs")
    print("🔧 Alternative docs at http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "tax_aware_api:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except ImportError:
        print("❌ Error: FastAPI or uvicorn not installed")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
