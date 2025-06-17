# 🎨 Enhanced AI Image Generator - Full Stack Web3 Application

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Web3](https://img.shields.io/badge/Web3-Enabled-purple.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

An end to end AI image generation platform leveraging dual Stable Diffusion architectures (ReV Animated for anime-style artwork, DreamShaper for photorealistic imagery) with GPU-accelerated inference, enterprise-grade AWS S3 cloud integration, intelligent content moderation systems, and native blockchain NFT creation on Polygon network. Developed using PyTorch framework, Gradio interface, and Web3 smart contract protocols to deliver end-to-end digital asset generation and cryptographic ownership verification.

---
## 🎥 Video Demo

https://github.com/user-attachments/assets/8543b145-58dd-4981-914d-73c41958620d


---

## ✨ Features

### 🤖 AI Generation Core
- **Multi-Model Support**: Switch between ReV Animated (anime/fantasy) and DreamShaper (realistic) models
- **Advanced Controls**: Fine-tune generation with steps, CFG scale, resolution, and seed control
- **Image-to-Image**: Transform existing images using AI-powered style transfer
- **Real-time Progress**: Live generation progress with step-by-step updates
- **GPU Acceleration**: Automatic CUDA detection for 20-50x faster generation

### 🛡️ Safety & Quality
- **NSFW Content Detection**: AI-powered inappropriate content filtering using transformers
- **Smart Prompt Enhancement**: Local LLM (Ollama) automatically improves prompts for better results
- **Safe Alternatives**: Automatic suggestion of safer prompts when content violations detected
- **Quality Modifiers**: Intelligent addition of quality tags for superior image generation

### ☁️ Cloud Integration
- **Dual Storage System**: Automatic backup to both local storage and AWS S3 cloud
- **Metadata Preservation**: Complete generation parameters embedded in PNG files
- **Free Tier Support**: Optimized for AWS free tier with 5GB storage
- **Cross-Platform Access**: Access your images from anywhere with cloud sync

### 🌐 Web3 & Blockchain
- **NFT Minting**: Convert your AI art into blockchain-verified NFTs
- **Polygon Integration**: Deploy on eco-friendly, low-cost Polygon network
- **IPFS Storage**: Decentralized storage for permanent, censorship-resistant hosting
- **MetaMask Ready**: Seamless wallet integration for Web3 interactions
- **Testnet Support**: Full demo functionality with free test tokens

### 🎨 User Experience
- **Modern Interface**: Clean, responsive Gradio-based web UI with dark/light themes
- **Drag & Drop**: Intuitive image upload with visual feedback
- **Keyboard Shortcuts**: Power-user features (Ctrl+Enter to generate, Ctrl+T for theme)
- **Auto-Save**: Session persistence and automatic form data recovery
- **Real-time Gallery**: Live updates of recent generations with fullscreen viewer

### 📊 Advanced Features
- **Generation History**: Complete tracking of all creations with searchable metadata
- **Preset Library**: Quick-start prompts for different artistic styles
- **Batch Processing**: Queue multiple generations efficiently
- **Performance Monitoring**: Real-time system status and resource usage
- **Error Recovery**: Graceful handling of failures with detailed diagnostics

---

## 📋 Prerequisites
- **Python 3.8** or higher (3.10+ recommended)
- **8GB RAM** minimum (16GB recommended for optimal performance)
- **10GB Storage** for models and generated images
- **NVIDIA GPU** (optional but recommended for 20-50x speed boost)
- **Ollama** for local LLM prompt enhancement
- **AWS Account** (free tier) for cloud storage
- **MetaMask Wallet** for Web3 features

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/enhanced-ai-image-generator.git
cd enhanced-ai-image-generator
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Downloads
```bash
# Create models directory
mkdir models

# Download models from Civitai:
# 1. ReV Animated: https://civitai.com/models/7371/rev-animated
# 2. DreamShaper: https://civitai.com/models/4384/dreamshaper
# Place .safetensors files in the models/ folder
```

### 4. Local LLM Setup (Ollama)
```bash
# Install Ollama from https://ollama.ai
# Pull the required model
ollama pull llama3.2:3b

# Verify Ollama is running
ollama list
```

### 5. AWS Cloud Storage Setup
```bash
# 1. Create AWS account (free tier)
# 2. Create S3 bucket via AWS Console
# 3. Generate IAM access keys
# 4. Configure environment variables
```

### 6. Environment Configuration
```bash
# Copy template and configure
cp .env.template .env

# Edit .env with your credentials:
# - AWS access keys and bucket name
# - Model file paths
# - Blockchain network settings
```

### 7. Launch Application
```bash
# Run the enhanced startup script
python start.py

# Or run directly
python app.py
```

The application will open automatically at http://127.0.0.1:7860

---

## 🎯 Usage Guide

### Basic Image Generation

#### Getting Started:
1. **Load Model**: Select ReV Animated for anime/fantasy or DreamShaper for realistic images
2. **Enter Prompt**: Describe your desired image or select from presets
3. **Safety Check**: Verify prompt passes content filtering
4. **Enhance**: Let AI improve your prompt for better results
5. **Generate**: Create your masterpiece with real-time progress

#### Advanced Controls:
- **Resolution**: 512x512 to 1024x1024 (higher = more detail, slower generation)
- **Steps**: 5-50 inference steps (more = better quality but slower)
- **CFG Scale**: 1-20 prompt adherence (higher = follows prompt more strictly)
- **Seed**: Control randomness for reproducible results

### Image-to-Image Transformation

#### Process:
1. **Enable img2img**: Toggle image-to-image mode
2. **Upload Reference**: Drag & drop or click to upload source image
3. **Set Strength**: 0.1-1.0 (lower = closer to original, higher = more creative)
4. **Add Prompt**: Describe desired changes or style
5. **Transform**: Generate variations based on your reference

### Cloud Storage & Sync

#### Automatic Features:
- **Dual Save**: Every image saved locally AND to AWS S3
- **Metadata Embedding**: Full generation parameters stored in PNG files
- **Gallery Sync**: Recent images automatically populate gallery
- **Cross-Device Access**: Access your creations from any device

### NFT Minting (Web3)

#### Blockchain Integration:
1. **Generate Art**: Create your AI masterpiece
2. **Connect Wallet**: Link MetaMask for blockchain interactions
3. **Mint NFT**: Convert image to blockchain-verified NFT
4. **Get Certificate**: Receive unique Token ID and transaction proof
5. **View on OpenSea**: See your NFT on decentralized marketplaces

#### What You Get:
- **Unique Token ID**: Permanent blockchain identifier
- **Transaction Hash**: Immutable proof of creation
- **IPFS Storage**: Decentralized, permanent image hosting
- **Marketplace Links**: Direct links to OpenSea and blockchain explorers
- **Ownership Proof**: Cryptographically verified authenticity

---

## 🔐 Configuration

### Environment Variables (.env)
```env
# AWS Cloud Storage (Free Tier)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-unique-bucket-name

# Local LLM Enhancement
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# AI Models
REV_ANIMATED_PATH=./models/revAnimated_v2Rebirth.safetensors
DREAMSHAPER_PATH=./models/dreamshaper_8.safetensors

# Blockchain (Testnet - Free)
NETWORK_NAME=mumbai
NETWORK_RPC=https://rpc-mumbai.maticvigil.com/
CHAIN_ID=80001
```

### Model Specifications
| Model | Specialty | Size | Best For |
|-------|-----------|------|----------|
| ReV Animated | Anime/Fantasy | 4GB | Characters, fantasy scenes, anime style |
| DreamShaper | Realistic/Artistic | 2GB | Photorealistic images, artistic concepts |

---

## 🚀 Performance Optimization

### Hardware Recommendations
| Component | Minimum | Recommended | Performance Impact |
|-----------|---------|-------------|-------------------|
| **CPU** | 4 cores | 8+ cores | Model loading speed |
| **RAM** | 8GB | 16GB+ | Stability & model switching |
| **GPU** | None (CPU) | RTX 3060+ | 20-50x faster generation |
| **Storage** | 20GB | 50GB+ SSD | Model loading & image storage |

### Generation Speed Comparison
| Hardware | Time per Image | Throughput |
|----------|---------------|------------|
| **CPU Only** | 3-5 minutes | ~0.3 images/hour |
| **RTX 3060** | 15-30 seconds | ~100 images/hour |
| **RTX 4090** | 5-10 seconds | ~300 images/hour |

### Optimization Tips
- **Use GPU**: Install CUDA-enabled PyTorch for massive speedup
- **Lower Steps**: Reduce to 15-20 for faster generation
- **Batch Processing**: Generate multiple images in sequence
- **Model Caching**: Keep frequently used models loaded

---

## 📊 Analytics & Monitoring

### Built-in Metrics
- **Generation Statistics**: Success rates, timing, model usage
- **Storage Analytics**: Local vs cloud usage, file sizes
- **Performance Monitoring**: GPU utilization, memory usage
- **Error Tracking**: Comprehensive logging and diagnostics

### System Status Dashboard
```
🚀 GPU: NVIDIA RTX 4090 (24.0GB)
🤖 Model: ReV Animated (loaded)
🛡️ NSFW Detection: Active
🧠 LLM: llama3.2:3b (ready)
💾 Storage: Local(✅) | Cloud(✅)
🔗 Blockchain: mumbai (✅)
```

---


## 🐛 Troubleshooting

### Common Issues

#### **Model Loading Errors**
```bash
# Check model files exist
ls -la models/

# Verify file paths in .env
cat .env | grep _PATH

# Check GPU memory
python -c "import torch; print(torch.cuda.memory_summary())"
```

#### **AWS Connection Issues**
```bash
# Test AWS credentials
aws s3 ls s3://your-bucket-name

# Verify IAM permissions
aws sts get-caller-identity
```

#### **Generation Hanging**
```bash
# Check GPU availability
nvidia-smi

# Monitor system resources
htop

# Check Gradio logs for errors
```

#### **Web3 Connection Failed**
```bash
# Verify network connectivity
curl https://rpc-mumbai.maticvigil.com/

# Check MetaMask network settings
# Ensure testnet is selected
```

---

## 🔭 Project Outlook


![Image](https://github.com/user-attachments/assets/db54a638-9fe7-46fd-802e-759e078e808e)
![Image](https://github.com/user-attachments/assets/f858d596-d51e-42a0-a582-dca5aad2a9fb)
![Image](https://github.com/user-attachments/assets/397b3597-ffc4-436a-aea0-0654531950c7)
![Image](https://github.com/user-attachments/assets/327071a6-650b-409c-ab3e-e51d446378e6)
![Image](https://github.com/user-attachments/assets/c46eb96e-7283-44f4-9897-4599069c5d3a)
![Image](https://github.com/user-attachments/assets/62600a19-c00f-43d2-a677-68c324ad201e)
![Image](https://github.com/user-attachments/assets/a6dff5bc-ac51-4e82-88c7-0cfd9509becb)
![Image](https://github.com/user-attachments/assets/54148c6c-7371-4a7a-84ef-71bf845090dd)

---

## 🙏 Acknowledgments

### Technology Partners
- **Stability AI** for Stable Diffusion models
- **Hugging Face** for the Diffusers library and model hosting
- **Civitai** community for high-quality model sharing
- **Gradio** for the intuitive web interface framework
- **Ollama** for local LLM integration
- **AWS** for reliable cloud infrastructure
- **Polygon** for eco-friendly blockchain infrastructure

