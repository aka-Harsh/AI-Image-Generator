import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Model Configurations
    REV_ANIMATED_PATH = os.getenv("REV_ANIMATED_PATH", "./models/revAnimated_v122.safetensors")
    DREAMSHAPER_PATH = os.getenv("DREAMSHAPER_PATH", "./models/dreamshaper_8.safetensors")
    
    # Storage Configurations
    OUTPUT_DIR = "./outputs"
    HISTORY_FILE = "./outputs/generation_history.json"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    
    # Blockchain Configuration (Testnet)
    NETWORK_NAME = os.getenv("NETWORK_NAME", "mumbai")
    NETWORK_RPC = os.getenv("NETWORK_RPC", "https://rpc-mumbai.maticvigil.com/")
    CHAIN_ID = int(os.getenv("CHAIN_ID", "80001"))
    
    # Generation Settings
    DEFAULT_NEGATIVE_PROMPT = "(worst quality, low quality:1.4), blurry, distorted, ugly, nsfw"
    
    # NSFW Detection Settings
    NSFW_THRESHOLD = 0.7  # Confidence threshold for blocking content
    
    # Model Presets
    AVAILABLE_MODELS = {
        "ReV Animated": {
            "path": REV_ANIMATED_PATH,
            "description": "Anime and fantasy art specialist",
            "recommended_resolution": (512, 512),
            "style": "anime"
        },
        "DreamShaper": {
            "path": DREAMSHAPER_PATH,
            "description": "Realistic and artistic images",
            "recommended_resolution": (512, 768),
            "style": "realistic"
        }
    }
    
    # Prompt Presets (Enhanced)
    PROMPT_PRESETS = {
        "Anime Portrait": "((best quality)), ((masterpiece)), (detailed), anime girl portrait, beautiful eyes, detailed face",
        "Fantasy Landscape": "((best quality)), ((masterpiece)), fantasy landscape, magical forest, detailed background, epic scenery",
        "Anime Warrior": "((best quality)), ((masterpiece)), anime warrior, detailed armor, weapon, action pose, dramatic lighting",
        "Cute Anime Girl": "((best quality)), ((masterpiece)), cute anime girl, kawaii, colorful clothes, happy expression",
        "Dark Fantasy": "((best quality)), ((masterpiece)), dark fantasy, gothic atmosphere, mysterious, detailed shadows",
        "Realistic Portrait": "((best quality)), ((masterpiece)), realistic portrait, detailed skin, professional lighting",
        "Cyberpunk City": "((best quality)), ((masterpiece)), cyberpunk cityscape, neon lights, futuristic, detailed architecture",
        "Nature Scene": "((best quality)), ((masterpiece)), beautiful nature, landscape, detailed trees, natural lighting"
    }
    
    # Style Modifiers
    STYLE_MODIFIERS = [
        "highly detailed", "8k resolution", "studio lighting", "sharp focus",
        "dramatic lighting", "vibrant colors", "soft lighting", "cinematic",
        "painterly", "digital art", "concept art", "illustration",
        "photorealistic", "hyperrealistic", "artstation", "trending"
    ]
    
    # Generation Limits
    MAX_RESOLUTION = 1024
    MIN_RESOLUTION = 256
    MAX_STEPS = 50
    MIN_STEPS = 5
    MAX_CFG_SCALE = 20
    MIN_CFG_SCALE = 1

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if model files exist
    if not os.path.exists(Config.REV_ANIMATED_PATH):
        errors.append(f"ReV Animated model not found: {Config.REV_ANIMATED_PATH}")
    
    if not os.path.exists(Config.DREAMSHAPER_PATH):
        errors.append(f"DreamShaper model not found: {Config.DREAMSHAPER_PATH}")
    
    # Check AWS credentials (but don't require CLI)
    if not Config.AWS_ACCESS_KEY_ID or not Config.AWS_SECRET_ACCESS_KEY:
        errors.append("AWS credentials not configured in .env file")
    elif Config.AWS_ACCESS_KEY_ID.startswith("your_") or Config.AWS_SECRET_ACCESS_KEY.startswith("your_"):
        errors.append("AWS credentials still contain placeholder values")
    
    # Test S3 connection (optional, don't fail if network issues)
    if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_REGION
            )
            
            # Test bucket access
            s3_client.head_bucket(Bucket=Config.S3_BUCKET_NAME)
            print("✅ S3 bucket accessible")
            
        except NoCredentialsError:
            errors.append("AWS credentials are invalid")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                errors.append(f"S3 bucket not found: {Config.S3_BUCKET_NAME}")
            elif error_code == '403':
                errors.append("Access denied to S3 bucket (check permissions)")
            else:
                print(f"⚠️ S3 connection warning: {e}")
        except Exception as e:
            print(f"⚠️ Could not test S3 connection: {e}")
    
    # Check Ollama connection (optional, don't fail if not running)
    try:
        import requests
        response = requests.get(f"{Config.OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if any(Config.OLLAMA_MODEL in m['name'] for m in models):
                print("✅ Ollama service running with required model")
            else:
                print(f"⚠️ Ollama running but {Config.OLLAMA_MODEL} not found")
                print(f"   Run: ollama pull {Config.OLLAMA_MODEL}")
        else:
            print("⚠️ Ollama service not responding")
    except:
        print("⚠️ Ollama service not running (prompt enhancement disabled)")
        print("   Start with: ollama serve")
    
    return errors

if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("❌ Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")