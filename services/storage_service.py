"""
Dual Storage Service
Handles both local and AWS S3 cloud storage for generated images
"""

import os
import boto3
import json
import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Dict, Optional, Tuple, List
from botocore.exceptions import ClientError, NoCredentialsError
from config import Config
import logging

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        """Initialize storage service with AWS S3 client"""
        self.s3_client = None
        self.bucket_name = Config.S3_BUCKET_NAME
        self.local_dir = Config.OUTPUT_DIR
        self.setup_local_storage()
        self.setup_s3_client()
    
    def setup_local_storage(self):
        """Create local output directory if it doesn't exist"""
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info(f"Local storage ready: {self.local_dir}")
    
    def setup_s3_client(self):
        """Initialize AWS S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_REGION
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.s3_client = None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket not found: {self.bucket_name}")
            else:
                logger.error(f"S3 connection error: {e}")
            self.s3_client = None
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if Config.AWS_REGION == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': Config.AWS_REGION}
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    return False
            else:
                logger.error(f"S3 bucket check error: {e}")
                return False
    
    def generate_filename(self, prompt: str, seed: int) -> str:
        """Generate filename based on timestamp and seed"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean prompt for filename (remove special characters)
        clean_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).strip()
        clean_prompt = clean_prompt.replace(' ', '_')
        
        return f"generated_{timestamp}_{seed}_{clean_prompt}.png"
    
    def save_image_metadata(self, image: Image.Image, metadata: Dict) -> Image.Image:
        """Add metadata to image as PNG info"""
        pnginfo = PngInfo()
        
        # Add generation parameters
        pnginfo.add_text("parameters", json.dumps(metadata))
        pnginfo.add_text("software", "Enhanced AI Image Generator")
        pnginfo.add_text("timestamp", datetime.datetime.now().isoformat())
        
        # Create new image with metadata
        img_with_metadata = Image.new(image.mode, image.size)
        img_with_metadata.paste(image)
        
        return img_with_metadata, pnginfo
    
    def save_locally(self, image: Image.Image, filename: str, metadata: Dict) -> Tuple[bool, str]:
        """Save image to local storage"""
        try:
            local_path = os.path.join(self.local_dir, filename)
            
            # Add metadata to image
            img_with_metadata, pnginfo = self.save_image_metadata(image, metadata)
            
            # Save image
            img_with_metadata.save(local_path, pnginfo=pnginfo)
            
            logger.info(f"Image saved locally: {local_path}")
            return True, local_path
            
        except Exception as e:
            logger.error(f"Failed to save image locally: {e}")
            return False, str(e)
    
    def upload_to_s3(self, local_path: str, s3_key: str) -> Tuple[bool, str]:
        """Upload image to S3"""
        if not self.s3_client:
            return False, "S3 client not available"
        
        try:
            # Upload file
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png',
                    'CacheControl': 'max-age=31536000',  # 1 year cache
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{Config.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"Image uploaded to S3: {s3_url}")
            return True, s3_url
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected S3 upload error: {e}")
            return False, str(e)
    
    def save_image_dual(self, image: Image.Image, metadata: Dict) -> Dict[str, any]:
        """
        Save image to both local and cloud storage
        Returns comprehensive result dictionary
        """
        # Generate filename
        prompt = metadata.get('prompt', 'unknown')
        seed = metadata.get('seed', 0)
        filename = self.generate_filename(prompt, seed)
        
        result = {
            'filename': filename,
            'local_success': False,
            'cloud_success': False,
            'local_path': None,
            'cloud_url': None,
            'errors': []
        }
        
        # Save locally first
        local_success, local_result = self.save_locally(image, filename, metadata)
        result['local_success'] = local_success
        
        if local_success:
            result['local_path'] = local_result
            
            # Try to upload to S3
            s3_key = f"generations/{filename}"
            cloud_success, cloud_result = self.upload_to_s3(local_result, s3_key)
            result['cloud_success'] = cloud_success
            
            if cloud_success:
                result['cloud_url'] = cloud_result
            else:
                result['errors'].append(f"Cloud upload failed: {cloud_result}")
        else:
            result['errors'].append(f"Local save failed: {local_result}")
        
        # Log summary
        status = []
        if result['local_success']:
            status.append("✅ Local")
        else:
            status.append("❌ Local")
            
        if result['cloud_success']:
            status.append("✅ Cloud")
        else:
            status.append("❌ Cloud")
        
        logger.info(f"Storage result: {' | '.join(status)} - {filename}")
        
        return result
    
    def get_recent_images(self, limit: int = 10) -> List[str]:
        """Get list of recent local images"""
        try:
            if not os.path.exists(self.local_dir):
                return []
            
            image_files = []
            for file in os.listdir(self.local_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(self.local_dir, file)
                    image_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: x[1], reverse=True)
            
            return [file_path for file_path, _ in image_files[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to get recent images: {e}")
            return []
    
    def get_storage_status(self) -> Dict[str, any]:
        """Get current storage system status"""
        status = {
            'local_available': os.path.exists(self.local_dir),
            'local_path': self.local_dir,
            's3_available': self.s3_client is not None,
            's3_bucket': self.bucket_name,
            's3_region': Config.AWS_REGION,
            'total_local_images': 0,
            'local_storage_size': 0
        }
        
        # Count local images
        try:
            if status['local_available']:
                for file in os.listdir(self.local_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        status['total_local_images'] += 1
                        file_path = os.path.join(self.local_dir, file)
                        status['local_storage_size'] += os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
        
        return status

# Global storage service instance
_storage_service = None

def get_storage_service() -> StorageService:
    """Get singleton storage service instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

# Test function
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Test metadata
    test_metadata = {
        'prompt': 'test image generation',
        'seed': 12345,
        'width': 512,
        'height': 512,
        'steps': 20,
        'cfg_scale': 7.5
    }
    
    # Test storage
    storage = StorageService()
    result = storage.save_image_dual(test_image, test_metadata)
    
    print("Storage Test Results:")
    print(f"Filename: {result['filename']}")
    print(f"Local Success: {result['local_success']}")
    print(f"Cloud Success: {result['cloud_success']}")
    print(f"Errors: {result['errors']}")
    
    print("\nStorage Status:")
    status = storage.get_storage_status()
    for key, value in status.items():
        print(f"{key}: {value}")