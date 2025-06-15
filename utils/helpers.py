"""
Utility helper functions for the Enhanced AI Image Generator
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def get_file_hash(file_path: str) -> Optional[str]:
    """Get SHA256 hash of a file"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Failed to get directory size for {path}: {e}")
    return total_size

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def load_json_file(file_path: str) -> Optional[Dict]:
    """Safely load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return None

def save_json_file(data: Dict, file_path: str) -> bool:
    """Safely save data to JSON file"""
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False

def get_image_info(image_path: str) -> Dict:
    """Get detailed information about an image file"""
    info = {
        "exists": False,
        "format": None,
        "size": None,
        "mode": None,
        "file_size": 0,
        "created": None,
        "modified": None
    }
    
    try:
        if os.path.exists(image_path):
            info["exists"] = True
            info["file_size"] = os.path.getsize(image_path)
            
            stat = os.stat(image_path)
            info["created"] = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
            info["modified"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            with Image.open(image_path) as img:
                info["format"] = img.format
                info["size"] = img.size
                info["mode"] = img.mode
                
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
    
    return info

def resize_image(image: Image.Image, max_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """Resize image while optionally maintaining aspect ratio"""
    if not maintain_aspect:
        return image.resize(max_size, Image.Resampling.LANCZOS)
    
    # Calculate size maintaining aspect ratio
    img_ratio = image.width / image.height
    max_ratio = max_size[0] / max_size[1]
    
    if img_ratio > max_ratio:
        # Image is wider
        new_width = max_size[0]
        new_height = int(max_size[0] / img_ratio)
    else:
        # Image is taller
        new_height = max_size[1]
        new_width = int(max_size[1] * img_ratio)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """Validate if file is a valid image"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, "Valid image file"
    except Exception as e:
        return False, f"Invalid image file: {e}"

def get_recent_files(directory: str, extension: str = None, limit: int = 10) -> List[Dict]:
    """Get list of recent files in directory"""
    files = []
    
    try:
        if not os.path.exists(directory):
            return files
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                if extension and not filename.lower().endswith(extension.lower()):
                    continue
                
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "path": file_path,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "created": stat.st_ctime
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return files[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get recent files from {directory}: {e}")
        return []

def cleanup_old_files(directory: str, max_files: int = 100, max_age_days: int = 30) -> int:
    """Clean up old files to maintain storage limits"""
    if not os.path.exists(directory):
        return 0
    
    deleted_count = 0
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
    
    try:
        files = get_recent_files(directory, limit=1000)  # Get more files for cleanup
        
        # Delete files older than cutoff
        for file_info in files:
            if datetime.datetime.fromtimestamp(file_info["modified"]) < cutoff_time:
                try:
                    os.remove(file_info["path"])
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_info['filename']}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_info['path']}: {e}")
        
        # If still too many files, delete oldest ones
        remaining_files = get_recent_files(directory, limit=1000)
        if len(remaining_files) > max_files:
            files_to_delete = remaining_files[max_files:]
            for file_info in files_to_delete:
                try:
                    os.remove(file_info["path"])
                    deleted_count += 1
                    logger.info(f"Deleted excess file: {file_info['filename']}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_info['path']}: {e}")
        
    except Exception as e:
        logger.error(f"Cleanup failed for {directory}: {e}")
    
    return deleted_count

def generate_unique_id() -> str:
    """Generate a unique ID based on timestamp and random data"""
    import uuid
    return str(uuid.uuid4()).replace('-', '')[:16]

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def extract_seed_from_filename(filename: str) -> Optional[int]:
    """Extract seed value from generated image filename"""
    try:
        # Expected format: generated_YYYYMMDD_HHMMSS_SEED_description.png
        parts = filename.split('_')
        if len(parts) >= 4:
            return int(parts[3])
    except (ValueError, IndexError):
        pass
    return None

def create_metadata_summary(metadata: Dict) -> str:
    """Create a human-readable summary of generation metadata"""
    summary_parts = []
    
    if "prompt" in metadata:
        prompt = metadata["prompt"][:50] + "..." if len(metadata["prompt"]) > 50 else metadata["prompt"]
        summary_parts.append(f"Prompt: {prompt}")
    
    if "model" in metadata:
        summary_parts.append(f"Model: {metadata['model']}")
    
    if "seed" in metadata:
        summary_parts.append(f"Seed: {metadata['seed']}")
    
    if "width" in metadata and "height" in metadata:
        summary_parts.append(f"Size: {metadata['width']}x{metadata['height']}")
    
    if "steps" in metadata:
        summary_parts.append(f"Steps: {metadata['steps']}")
    
    if "cfg_scale" in metadata:
        summary_parts.append(f"CFG: {metadata['cfg_scale']}")
    
    return " | ".join(summary_parts)

def validate_system_requirements() -> List[str]:
    """Check system requirements and return list of issues"""
    issues = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (current: {sys.version_info.major}.{sys.version_info.minor})")
    
    # Check available disk space
    import shutil
    free_space = shutil.disk_usage('.').free
    if free_space < 5 * 1024**3:  # 5GB
        issues.append(f"Low disk space: {format_file_size(free_space)} available")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 6 * 1024**3:  # 6GB
            issues.append(f"Low RAM: {format_file_size(memory.total)} available")
    except ImportError:
        issues.append("Cannot check memory (psutil not installed)")
    
    return issues

# Test functions
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test directory operations
    test_dir = "./test_utils"
    print(f"Creating directory: {ensure_directory(test_dir)}")
    
    # Test JSON operations
    test_data = {"test": "data", "timestamp": datetime.datetime.now().isoformat()}
    json_file = os.path.join(test_dir, "test.json")
    print(f"Saving JSON: {save_json_file(test_data, json_file)}")
    loaded_data = load_json_file(json_file)
    print(f"Loading JSON: {loaded_data == test_data}")
    
    # Test file size formatting
    print(f"Format size: {format_file_size(1234567890)}")
    
    # Test system requirements
    issues = validate_system_requirements()
    if issues:
        print("System issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("System requirements OK")
    
    # Cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Utility tests completed!")