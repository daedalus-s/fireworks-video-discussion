import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_setup():
    print("🔍 Testing Vector Search Setup...")
    
    # Check API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return False
    print("✅ Pinecone API key found")
    
    # Test imports
    try:
        import pinecone
        print("✅ Pinecone library imported")
    except ImportError:
        print("❌ Pinecone not installed. Run: pip install pinecone-client")
        return False
    
    try:
        import sentence_transformers
        print("✅ SentenceTransformers library imported")
    except ImportError:
        print("❌ SentenceTransformers not installed. Run: pip install sentence-transformers")
        return False
    
    # Test Pinecone connection
    try:
        from vector_search_system import VideoSearchInterface
        search = VideoSearchInterface()
        print("✅ Vector search system initialized successfully")
        
        # Get stats
        stats = search.search_stats()
        print(f"📊 Pinecone index stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    if test_setup():
        print("\n🎉 Setup complete! Vector search is ready to use.")
    else:
        print("\n💥 Setup failed. Check the errors above.")