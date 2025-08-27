#!/usr/bin/env python3
"""
FIXED Test Script - Properly loads .env file
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# IMPORTANT: Load .env file first!
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, trying to read .env manually")
    # Manually read .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Manually loaded .env file")
    else:
        print("❌ No .env file found")

def test_module_imports():
    """Test that all your modules can be imported successfully"""
    
    print("\n🧪 Testing module imports...")
    print("=" * 50)
    
    modules_to_test = [
        "video_processor",
        "fireworks_client", 
        "video_analysis_system",
        "multi_agent_discussion",
        "configurable_agent_system",
        "enhanced_descriptive_analysis",
        "rag_enhanced_vector_system",
        "query_video_rag",
        "integrated_configurable_pipeline",
        "api_manager",
        "vector_search_system"
    ]
    
    success_count = 0
    failed_modules = []
    
    for module_name in modules_to_test:
        try:
            print(f"Testing {module_name}...", end=" ")
            
            if module_name == "video_processor":
                from video_processor import VideoProcessor, SubtitleProcessor
                print("✅ SUCCESS")
            elif module_name == "fireworks_client":
                from fireworks_client import FireworksClient
                print("✅ SUCCESS")
            elif module_name == "video_analysis_system":
                from video_analysis_system import VideoAnalysisSystem
                print("✅ SUCCESS")
            elif module_name == "multi_agent_discussion":
                from multi_agent_discussion import MultiAgentDiscussion
                print("✅ SUCCESS")
            elif module_name == "configurable_agent_system":
                from configurable_agent_system import ConfigurableMultiAgentDiscussion
                print("✅ SUCCESS")
            elif module_name == "enhanced_descriptive_analysis":
                from enhanced_descriptive_analysis import analyze_video_highly_descriptive
                print("✅ SUCCESS")
            elif module_name == "rag_enhanced_vector_system":
                from rag_enhanced_vector_system import RAGQueryInterface
                print("✅ SUCCESS")
            elif module_name == "query_video_rag":
                import query_video_rag
                print("✅ SUCCESS")
            elif module_name == "integrated_configurable_pipeline":
                from integrated_configurable_pipeline import IntegratedConfigurableAnalysisPipeline
                print("✅ SUCCESS")
            elif module_name == "api_manager":
                from api_manager import OptimizedAPIManager
                print("✅ SUCCESS")
            elif module_name == "vector_search_system":
                from vector_search_system import VideoSearchInterface
                print("✅ SUCCESS")
            
            success_count += 1
            
        except ImportError as e:
            print(f"❌ FAILED - ImportError: {e}")
            failed_modules.append((module_name, str(e)))
        except Exception as e:
            print(f"❌ FAILED - Error: {e}")
            failed_modules.append((module_name, str(e)))
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {success_count}/{len(modules_to_test)} modules imported successfully")
    
    if failed_modules:
        print(f"\n❌ Failed modules:")
        for module_name, error in failed_modules:
            print(f"  - {module_name}: {error}")
        print(f"\n💡 Next steps:")
        print(f"  1. Install missing dependencies: pip install -r requirements.txt")
        print(f"  2. Check for any missing dependencies specific to your setup")
    else:
        print(f"\n🎉 All modules imported successfully!")
        print(f"✅ Your existing code structure is working correctly")
    
    return len(failed_modules) == 0

def test_environment_variables():
    """Test that required environment variables are set"""
    print(f"\n🔑 Testing environment variables...")
    print("=" * 50)
    
    required_vars = ["FIREWORKS_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    # Debug: show all env vars that start with the key names
    print("Debug: Environment variables found:")
    for key in os.environ:
        if any(var_name in key for var_name in required_vars):
            value = os.environ[key]
            print(f"  {key} = {value[:10]}...{value[-5:] if len(value) > 15 else value}")
    
    for var in required_vars:
        value = os.getenv(var)
        print(f"Checking {var}...", end=" ")
        if value and len(value) > 10:
            print(f"✅ SET (length: {len(value)})")
        else:
            print(f"❌ NOT SET (value: '{value}')")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print(f"\n💡 Make sure your .env file contains:")
        print(f"     FIREWORKS_API_KEY=fw_your_key_here")
        print(f"     PINECONE_API_KEY=pcsk_your_key_here")
        return False
    else:
        print(f"\n✅ All required environment variables are set correctly!")
        return True

def test_directory_structure():
    """Test that all required directories exist"""
    print(f"\n📁 Testing directory structure...")
    print("=" * 50)
    
    required_dirs = ["static", "uploads", "results", "logs"]
    existing_dirs = ["frames", "real_videos", "test_videos"]
    
    # Check existing directories
    for dir_name in existing_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/: Exists (found)")
        else:
            print(f"⚠️  {dir_name}/: Expected but not found")
    
    # Check/create required directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/: Exists")
        else:
            print(f"❌ {dir_name}/: Missing")
            dir_path.mkdir(exist_ok=True)
            print(f"   📁 Created {dir_name}/ directory")
    
    return True

def test_video_files():
    """Check if you have test video files"""
    print(f"\n🎥 Checking for test video files...")
    print("=" * 50)
    
    video_dirs = ["real_videos", "test_videos"]
    video_files = []
    
    for dir_name in video_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                video_files.extend(list(dir_path.glob(ext)))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video files:")
        for video in video_files[:3]:
            print(f"  - {video}")
        if len(video_files) > 3:
            print(f"  ... and {len(video_files) - 3} more")
    else:
        print(f"⚠️  No video files found")
        print(f"💡 You can test with sample videos later")
    
    return len(video_files) > 0

def main():
    """Run all tests"""
    print("🎬 AI Video Analysis Platform - FIXED Step 1 Module Test")
    print("=" * 70)
    
    # Test directory structure
    dirs_ok = test_directory_structure()
    
    # Test environment variables (FIXED)
    env_ok = test_environment_variables()
    
    # Test module imports  
    modules_ok = test_module_imports()
    
    # Check video files
    videos_exist = test_video_files()
    
    print("\n" + "=" * 70)
    print("📋 STEP 1 FINAL SUMMARY")
    print("=" * 70)
    
    if modules_ok and dirs_ok and env_ok:
        print("🎉 PERFECT! Everything is set up correctly!")
        print("✅ All modules imported successfully")
        print("✅ Directory structure is correct")
        print("✅ Environment variables are configured properly")
        
        if videos_exist:
            print("✅ Test video files available")
        else:
            print("ℹ️  No test videos found (can add later)")
        
        print(f"\n🚀 READY FOR STEP 2: Setting up the FastAPI backend!")
        print(f"   Run: python main.py")
        
    else:
        print("❌ Issues found:")
        if not modules_ok:
            print("  - Some modules failed to import")
        if not dirs_ok:
            print("  - Directory structure issues")
        if not env_ok:
            print("  - Environment variables not set properly")
        
        print(f"\n🔧 Fix these issues before proceeding to Step 2")
    
    return modules_ok and dirs_ok and env_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)