#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the Moroccan income prediction API locally.

Usage:
    python run_api.py [--port PORT] [--host HOST]

Options:
    --port PORT    Port to run the API on [default: 8000]
    --host HOST    Host to run the API on [default: 127.0.0.1]
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Moroccan income prediction API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the API on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser.parse_args()

def main():
    """Run the API."""
    args = parse_args()
    
    print(f"Starting Moroccan Income Prediction API on http://{args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "api.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
