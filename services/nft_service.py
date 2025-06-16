"""
NFT Service for Web3 Integration
Handles NFT minting on Polygon testnet
"""

import json
import requests
from web3 import Web3
from eth_account import Account
from typing import Dict, Optional, Tuple
import hashlib
import datetime
from config import Config
import logging

logger = logging.getLogger(__name__)

class NFTService:
    def __init__(self):
        """Initialize NFT service with Web3 connection"""
        self.w3 = None
        self.contract = None
        self.contract_address = None
        self.setup_web3()
        self.setup_contract()
    
    def setup_web3(self):
        """Setup Web3 connection to testnet"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(Config.NETWORK_RPC))
            
            if self.w3.is_connected():
                logger.info(f"Connected to {Config.NETWORK_NAME} network")
                logger.info(f"Chain ID: {self.w3.eth.chain_id}")
            else:
                logger.error("Failed to connect to blockchain network")
                
        except Exception as e:
            logger.error(f"Web3 setup failed: {e}")
            self.w3 = None
    
    def setup_contract(self):
        """Setup NFT contract (we'll use a simple ERC721 for demo)"""
        # Simple ERC721 contract ABI (minimal for minting)
        self.contract_abi = [
            {
                "inputs": [
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "string", "name": "tokenURI", "type": "string"}
                ],
                "name": "mint",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"internalType": "string", "name": "", "type": "string"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # For demo purposes - you would deploy your own contract
        # This is a placeholder address
        self.contract_address = "0x0000000000000000000000000000000000000000"
        
        if self.w3 and self.w3.is_connected():
            try:
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=self.contract_abi
                )
                logger.info("NFT contract setup completed")
            except Exception as e:
                logger.warning(f"Contract setup failed: {e}")
                self.contract = None
    
    def upload_to_ipfs(self, image_data: bytes, metadata: Dict) -> Optional[str]:
        """
        Upload image and metadata to IPFS
        For demo, we'll simulate this with a hash-based approach
        In production, use services like Pinata, Infura IPFS, etc.
        """
        try:
            # Create a mock IPFS hash based on image data
            image_hash = hashlib.sha256(image_data).hexdigest()
            ipfs_hash = f"QmDemo{image_hash[:40]}"  # Mock IPFS hash
            
            # In real implementation, you would:
            # 1. Upload image to IPFS
            # 2. Create metadata JSON with image IPFS URL
            # 3. Upload metadata JSON to IPFS
            # 4. Return metadata IPFS hash
            
            mock_metadata = {
                "name": metadata.get("name", "AI Generated Art"),
                "description": metadata.get("description", "Generated using Enhanced AI Image Generator"),
                "image": f"ipfs://{ipfs_hash}",
                "attributes": [
                    {"trait_type": "Prompt", "value": metadata.get("prompt", "Unknown")},
                    {"trait_type": "Model", "value": metadata.get("model", "Unknown")},
                    {"trait_type": "Seed", "value": str(metadata.get("seed", 0))},
                    {"trait_type": "Steps", "value": str(metadata.get("steps", 0))},
                    {"trait_type": "CFG Scale", "value": str(metadata.get("cfg_scale", 0))},
                    {"trait_type": "Resolution", "value": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"},
                    {"trait_type": "Created", "value": datetime.datetime.now().isoformat()}
                ]
            }
            
            # Mock metadata upload
            metadata_hash = hashlib.sha256(json.dumps(mock_metadata).encode()).hexdigest()
            metadata_ipfs_hash = f"QmMeta{metadata_hash[:40]}"
            
            logger.info(f"Mock IPFS upload - Image: {ipfs_hash}, Metadata: {metadata_ipfs_hash}")
            return metadata_ipfs_hash
            
        except Exception as e:
            logger.error(f"IPFS upload simulation failed: {e}")
            return None
    
    def create_nft_metadata(self, generation_data: Dict) -> Dict:
        """Create NFT metadata from generation data"""
        return {
            "name": f"AI Art #{generation_data.get('seed', 'Unknown')}",
            "description": f"AI-generated artwork created with prompt: '{generation_data.get('prompt', 'Unknown')}'",
            "prompt": generation_data.get("prompt", ""),
            "model": generation_data.get("model", "Unknown"),
            "seed": generation_data.get("seed", 0),
            "steps": generation_data.get("steps", 0),
            "cfg_scale": generation_data.get("cfg_scale", 0),
            "width": generation_data.get("width", 0),
            "height": generation_data.get("height", 0),
            "created_at": datetime.datetime.now().isoformat()
        }
    
    def simulate_mint_nft(self, image_data: bytes, generation_data: Dict, user_address: str = None) -> Dict[str, any]:
        """
        Simulate NFT minting process for demo purposes
        In production, this would interact with real blockchain
        """
        try:
            # Create metadata
            metadata = self.create_nft_metadata(generation_data)
            
            # Simulate IPFS upload
            ipfs_hash = self.upload_to_ipfs(image_data, metadata)
            
            if not ipfs_hash:
                return {
                    "success": False,
                    "error": "Failed to upload to IPFS"
                }
            
            # Generate mock transaction data
            mock_tx_hash = f"0x{hashlib.sha256(f'{ipfs_hash}{datetime.datetime.now()}'.encode()).hexdigest()}"
            mock_token_id = abs(hash(ipfs_hash)) % 1000000  # Generate token ID
            
            # Simulate blockchain transaction
            mock_nft_data = {
                "success": True,
                "transaction_hash": mock_tx_hash,
                "token_id": mock_token_id,
                "contract_address": self.contract_address,
                "ipfs_hash": ipfs_hash,
                "metadata": metadata,
                "network": Config.NETWORK_NAME,
                "chain_id": Config.CHAIN_ID,
                "opensea_url": f"https://testnets.opensea.io/assets/{Config.NETWORK_NAME}/{self.contract_address}/{mock_token_id}",
                "etherscan_url": f"https://{Config.NETWORK_NAME}.etherscan.io/tx/{mock_tx_hash}",
                "estimated_gas": "0.001 MATIC",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"NFT minting simulated - Token ID: {mock_token_id}")
            return mock_nft_data
            
        except Exception as e:
            logger.error(f"NFT minting simulation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_nft_info(self, token_id: int) -> Dict[str, any]:
        """Get information about a specific NFT (simulated)"""
        return {
            "token_id": token_id,
            "contract_address": self.contract_address,
            "network": Config.NETWORK_NAME,
            "status": "Minted (Demo)",
            "opensea_url": f"https://testnets.opensea.io/assets/{Config.NETWORK_NAME}/{self.contract_address}/{token_id}",
            "note": "This is a demo implementation. In production, this would query the actual blockchain."
        }
    
    def validate_wallet_address(self, address: str) -> bool:
        """Validate Ethereum wallet address"""
        try:
            return Web3.is_address(address)
        except:
            return False
    
    def get_network_status(self) -> Dict[str, any]:
        """Get blockchain network status"""
        status = {
            "connected": False,
            "network": Config.NETWORK_NAME,
            "rpc_url": Config.NETWORK_RPC,
            "chain_id": Config.CHAIN_ID,
            "block_number": None,
            "gas_price": None
        }
        
        if self.w3 and self.w3.is_connected():
            try:
                status["connected"] = True
                status["block_number"] = self.w3.eth.block_number
                status["gas_price"] = self.w3.eth.gas_price
            except Exception as e:
                logger.error(f"Error getting network status: {e}")
        
        return status
    
    def estimate_gas_cost(self) -> str:
        """Estimate gas cost for NFT minting"""
        # Mock estimation for testnet
        return "~0.001 MATIC (Testnet - Free)"
    
    def get_contract_info(self) -> Dict[str, any]:
        """Get NFT contract information"""
        return {
            "address": self.contract_address,
            "network": Config.NETWORK_NAME,
            "type": "ERC721",
            "name": "AI Generated Art Collection",
            "symbol": "AIGA",
            "demo_note": "This is a demo contract address. In production, deploy your own NFT contract."
        }

# Global NFT service instance
_nft_service = None

def get_nft_service() -> NFTService:
    """Get singleton NFT service instance"""
    global _nft_service
    if _nft_service is None:
        _nft_service = NFTService()
    return _nft_service

def mint_nft_demo(image_data: bytes, generation_data: Dict) -> Dict[str, any]:
    """Convenience function to mint NFT (demo)"""
    nft_service = get_nft_service()
    return nft_service.simulate_mint_nft(image_data, generation_data)

# Test function
if __name__ == "__main__":
    nft_service = NFTService()
    
    print("NFT Service Test:")
    print(f"Network Status: {nft_service.get_network_status()}")
    print(f"Contract Info: {nft_service.get_contract_info()}")
    print(f"Gas Estimate: {nft_service.estimate_gas_cost()}")
    
    # Test NFT minting simulation
    test_generation_data = {
        "prompt": "test anime girl portrait",
        "model": "ReV Animated",
        "seed": 12345,
        "steps": 25,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512
    }
    
    # Mock image data
    test_image_data = b"mock_image_data_for_testing"
    
    result = nft_service.simulate_mint_nft(test_image_data, test_generation_data)
    print("\nNFT Minting Simulation:")
    for key, value in result.items():
        print(f"{key}: {value}")