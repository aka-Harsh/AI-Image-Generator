#!/usr/bin/env python3
"""
Enhanced AI Image Generator - Easy Startup Script
Handles system checks, service validation, and graceful startup
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🎨 Enhanced AI Image Generator 🎨                          ║
║                                                               ║
║    Multi-Model | NSFW Detection | Cloud Storage | NFT Ready  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

def print_banner():
    """Print startup banner"""
    print("\033[96m" + BANNER + "\033[0m")

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",     # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",    # Red
        "reset": "\033[0m"      # Reset
    }
    
    symbols = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    color = colors.get(status, colors["info"])
    symbol = symbols.get(status, "•")
    
    print(f"{color}{symbol} {message}{colors['reset']}")

def check_python_version():
    """Check Python version compatibility"""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status(f"Python 3.8+ required. Current: {version.major}.{version.minor}.{version.micro}", "error")
        return False
    
    print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "success")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    print_status("Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment detected ✓", "success")
        return True
    else:
        print_status("Not running in virtual environment", "warning")
        print_status("Recommendation: Use 'python -m venv venv' and activate it", "info")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_status("Checking dependencies...")
    
    required_packages = [
        'torch', 'diffusers', 'gradio', 'transformers', 
        'PIL', 'requests', 'boto3', 'web3', 'detoxify'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"  {package} ✓", "info")
        except ImportError:
            missing_packages.append(package)
            print_status(f"  {package} ❌", "error")
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "error")
        print_status("Run: pip install -r requirements.txt", "info")
        return False
    
    print_status("All dependencies installed ✓", "success")
    return True

def check_gpu_availability():
    """Check GPU availability for acceleration"""
    print_status("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_status(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB) ✓", "success")
            return True
        else:
            print_status("No GPU detected - will use CPU (slower)", "warning")
            return False
    except ImportError:
        print_status("PyTorch not installed - cannot check GPU", "error")
        return False

def check_ollama_service():
    """Check if Ollama is running"""
    print_status("Checking Ollama service...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print_status(f"Ollama running with {len(models)} models ✓", "success")
            if 'llama3.2:3b' in model_names:
                print_status("  llama3.2:3b model available ✓", "info")
            else:
                print_status("  llama3.2:3b model not found", "warning")
                print_status("  Run: ollama pull llama3.2:3b", "info")
            return True
        else:
            print_status("Ollama not responding", "error")
            return False
    except Exception as e:
        print_status("Ollama not running - prompt enhancement disabled", "warning")
        print_status("Start with: ollama serve", "info")
        return False

def check_model_files():
    """Check if AI model files are present"""
    print_status("Checking AI model files...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print_status("Models directory not found", "error")
        print_status("Create 'models/' folder and download model files", "info")
        return False
    
    model_files = list(models_dir.glob("*.safetensors"))
    if not model_files:
        print_status("No .safetensors model files found", "error")
        print_status("Download models from Civitai and place in models/ folder", "info")
        return False
    
    print_status(f"Found {len(model_files)} model file(s) ✓", "success")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024*1024)
        print_status(f"  {model_file.name} ({size_mb:.0f}MB)", "info")
    
    return True

def check_environment_config():
    """Check environment configuration"""
    print_status("Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print_status(".env file not found", "warning")
        print_status("Copy .env.template to .env and configure", "info")
        return False
    
    # Check for basic required variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET_NAME']
    missing_vars = []
    
    for var in required_vars:
        if f"{var}=your_" in env_content or f"{var}=" not in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print_status(f"Incomplete .env config: {', '.join(missing_vars)}", "warning")
        print_status("AWS features may not work", "info")
        return False
    
    print_status("Environment configuration looks good ✓", "success")
    return True

def check_disk_space():
    """Check available disk space"""
    print_status("Checking disk space...")
    
    import shutil
    free_space = shutil.disk_usage('.').free
    free_gb = free_space / (1024**3)
    
    if free_gb < 5:
        print_status(f"Low disk space: {free_gb:.1f}GB available", "error")
        print_status("At least 5GB recommended for model storage", "info")
        return False
    elif free_gb < 10:
        print_status(f"Adequate disk space: {free_gb:.1f}GB available", "warning")
        return True
    else:
        print_status(f"Sufficient disk space: {free_gb:.1f}GB available ✓", "success")
        return True

def start_application():
    """Start the main application"""
    print_status("Starting Enhanced AI Image Generator...", "info")
    print()
    
    try:
        # Import and run the main app
        import app
        print_status("Application started successfully! 🚀", "success")
        print_status("Open http://127.0.0.1:7860 in your browser", "info")
    except ImportError as e:
        print_status(f"Failed to import app.py: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Failed to start application: {e}", "error")
        return False
    
    return True

def show_troubleshooting_tips():
    """Show common troubleshooting tips"""
    print()
    print_status("Troubleshooting Tips:", "info")
    print("  • Install missing dependencies: pip install -r requirements.txt")
    print("  • Download models from Civitai and place in models/ folder")
    print("  • Start Ollama: ollama serve (in separate terminal)")
    print("  • Pull LLM model: ollama pull llama3.2:3b")
    print("  • Configure AWS: copy .env.template to .env and edit")
    print("  • Check GitHub README for detailed setup instructions")
    print()

def main():
    """Main startup sequence"""
    print_banner()
    print_status("Initializing Enhanced AI Image Generator...", "info")
    print()
    
    # System checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("GPU Availability", check_gpu_availability),
        ("Ollama Service", check_ollama_service),
        ("Model Files", check_model_files),
        ("Environment Config", check_environment_config),
        ("Disk Space", check_disk_space),
    ]
    
    critical_failures = []
    warnings = []
    
    print_status("Running system checks...", "info")
    print()
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result and check_name in ["Python Version", "Dependencies", "Model Files"]:
                critical_failures.append(check_name)
            elif not result:
                warnings.append(check_name)
        except Exception as e:
            print_status(f"Error during {check_name} check: {e}", "error")
            critical_failures.append(check_name)
        
        print()  # Add spacing between checks
    
    # Summary
    print("=" * 60)
    
    if critical_failures:
        print_status("Critical issues found - cannot start application:", "error")
        for failure in critical_failures:
            print_status(f"  ❌ {failure}", "error")
        print()
        show_troubleshooting_tips()
        return 1
    
    if warnings:
        print_status("Warnings (application may have limited functionality):", "warning")
        for warning in warnings:
            print_status(f"  ⚠️ {warning}", "warning")
        print()
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print_status("Startup cancelled by user", "info")
            return 0
        print()
    
    print_status("All checks passed! Starting application...", "success")
    print()
    
    # Start the application
    if start_application():
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print()
        print_status("Startup interrupted by user", "info")
        sys.exit(0)
    except Exception as e:
        print()
        print_status(f"Unexpected error during startup: {e}", "error")
        show_troubleshooting_tips()
        sys.exit(1)