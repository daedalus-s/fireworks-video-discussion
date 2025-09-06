# 🎬 AI Video Analysis Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-red.svg)](https://opencv.org/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple.svg)](https://www.pinecone.io/)

A comprehensive AI-powered video analysis platform that uses multiple specialized AI agents to analyze video content, extract insights, and enable advanced semantic search capabilities through RAG (Retrieval-Augmented Generation) and vector embeddings.

## 🌟 Key Features

- **🤖 Multi-Agent AI Analysis** - 4 specialized AI agents with distinct personalities and expertise
- **🔍 Semantic Search** - Natural language queries with temporal and agent-specific context
- **⚙️ Configurable Agents** - Customizable agent templates for different content types
- **🧠 RAG Integration** - Retrieval-augmented generation for intelligent responses
- **⏱️ Real-time Processing** - Live progress updates and background task management
- **🎬 Comprehensive Analysis** - Scene breakdown, visual elements, narrative structure
- **🔗 API-First Design** - RESTful FastAPI backend with modern web interface

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │───▶│   FastAPI API    │───▶│  Video Processor │
│   (HTML/JS/CSS) │    │   Background     │    │   (OpenCV)      │
└─────────────────┘    │   Tasks          │    └─────────────────┘
                       └──────────────────┘              │
                                │                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Pinecone       │◀───│   RAG System     │◀───│  AI Agents      │
│  Vector DB      │    │   (Semantic      │    │  (Fireworks.ai) │
└─────────────────┘    │   Search)        │    └─────────────────┘
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Fireworks.ai API key
- Pinecone API key (for RAG features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/daedalus-s/fireworks-video-discussion.git
cd fireworks-video-discussion
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Add your API keys
FIREWORKS_API_KEY=your_fireworks_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

4. **Run the application**
```bash
python main.py
```

5. **Open your browser**
Navigate to `http://localhost:8000` to access the web interface.

## 📖 Usage

### Basic Video Analysis

1. **Upload Video**: Drag and drop your video file (MP4, AVI, MOV)
2. **Configure Analysis**: 
   - Select analysis depth (Basic, Detailed, Comprehensive)
   - Choose AI agents (Alex, Maya, Jordan, Affan)
   - Set frame count and discussion rounds
3. **Start Analysis**: Click "Start Analysis" and watch real-time progress
4. **View Results**: Explore comprehensive analysis with scene breakdown
5. **Search Content**: Use natural language to query your analyzed content

### Advanced Features

#### Agent Templates
Choose from predefined agent configurations:
- **🎬 Film Analysis**: Cinematographer, Film Critic, Sound Designer
- **🎓 Educational**: Learning Specialist, Subject Expert, Engagement Analyst
- **📈 Marketing**: Brand Strategist, Conversion Specialist, Creative Director

#### Semantic Search Examples
```
"What did Alex say about the lighting?"
"Show me scenes with dramatic camera work"
"When does the character appear?"
"How did Maya interpret the emotions?"
"Find frames with multiple people"
```

#### API Usage
```python
import requests

# Upload video
files = {'video': open('video.mp4', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)
upload_id = response.json()['upload_id']

# Start analysis
config = {
    'analysis_depth': 'comprehensive',
    'max_frames': 10,
    'selected_agents': ['alex', 'maya', 'jordan']
}
response = requests.post(f'http://localhost:8000/api/analyze/{upload_id}', json=config)
task_id = response.json()['task_id']

# Check status
response = requests.get(f'http://localhost:8000/api/status/{task_id}')
results = response.json()
```

## 🤖 AI Agents

### Default Agents

| Agent | Role | Expertise | AI Model |
|-------|------|-----------|----------|
| **🎬 Alex** | Technical Analyst | Cinematography, Production Quality | GPT-OSS-120B |
| **🎨 Maya** | Creative Interpreter | Themes, Emotions, Artistic Analysis | Qwen3-235B |
| **👥 Jordan** | Audience Advocate | Engagement, Accessibility, UX | Llama4 Maverick |
| **💼 Affan** | Financial Analyst | Commercial Viability, Market Impact | GPT-OSS-120B |

### Custom Agent Configuration

Create custom agents by modifying `agent_configurations.json`:

```json
{
  "name": "Custom Agent",
  "role": "Domain Expert",
  "personality": "Analytical and detail-oriented",
  "expertise": ["domain_knowledge", "analysis"],
  "model": "gpt_oss",
  "emoji": "🔬"
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
FIREWORKS_API_KEY=your_fireworks_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional
PINECONE_ENVIRONMENT=us-east-1-aws
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=500MB
```

### Analysis Settings

- **Analysis Depth**: `basic`, `detailed`, `comprehensive`
- **Max Frames**: 5-50 frames (impacts cost and processing time)
- **FPS Extract**: 0.1-1.0 (frames per second to extract)
- **Discussion Rounds**: 1-5 rounds of agent discussion

## 📊 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/api/health` | System health check |
| `POST` | `/api/upload` | Upload video file |
| `POST` | `/api/analyze/{upload_id}` | Start analysis |
| `GET` | `/api/status/{task_id}` | Get analysis status |
| `POST` | `/api/search` | Semantic search |
| `GET` | `/api/agents` | List available agents |
| `GET` | `/api/tasks` | List active tasks |

### Response Format

```json
{
  "video_path": "uploads/video.mp4",
  "analysis_summary": {
    "frames_analyzed": 10,
    "agents_participated": 3,
    "total_cost": 0.0247
  },
  "key_insights": {
    "comprehensive_assessment": "...",
    "scene_breakdown": [...],
    "visual_highlights": [...]
  }
}
```

## 🧠 RAG & Search System

### Vector Search Features

- **Semantic Search**: Natural language queries
- **Temporal Queries**: Time-based content retrieval
- **Agent-Specific Search**: Query specific agent perspectives
- **Frame-Level Search**: Find specific visual moments

### Search Query Types

```python
# Agent-specific queries
"What did Alex analyze about camera techniques?"

# Temporal queries  
"What happens at 2:30 in the video?"
"Show me frame 15"

# Content-based queries
"Find scenes with dramatic lighting"
"Locate moments with emotional expressions"

# Cross-referencing queries
"When did multiple agents agree on something?"
```

## 📁 Project Structure

```
ai-video-analysis-platform/
├── 📄 main.py                              # FastAPI backend server
├── 🎬 video_processor.py                   # Video frame extraction
├── 🔥 fireworks_client.py                  # AI service integration
├── 🤖 configurable_agent_system.py         # Multi-agent framework
├── 🔬 enhanced_descriptive_analysis.py     # Advanced analysis
├── 🧠 rag_enhanced_vector_system.py        # RAG and vector search
├── 🔍 vector_search_system.py              # Semantic search
├── ⏰ universal_temporal_processor.py      # Temporal query processing
├── 🔗 integrated_configurable_pipeline.py # End-to-end pipeline
├── 🌐 integrated_rag_pipeline.py           # RAG workflow
├── 💬 multi_agent_discussion.py            # Agent discussions
├── 📊 video_analysis_system.py             # Core analysis
├── 🎯 query_video_rag.py                   # Query interface
├── ⚡ api_manager.py                        # Rate limiting
├── 📋 agent_configurations.json            # Agent definitions
├── 🖥️ static/
│   └── index.html                          # Web interface
├── 📦 requirements.txt                     # Dependencies
└── 📖 README.md                            # This file
```

## 🔧 Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Start development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running Individual Components

```bash
# Test video processor
python video_processor.py

# Test AI client
python fireworks_client.py

# Test agent system
python configurable_agent_system.py configure

# Test RAG system
python rag_enhanced_vector_system.py test
```

### Adding Custom AI Models

1. Extend `fireworks_client.py` with new model definitions
2. Update agent configurations to use new models
3. Modify rate limiting settings in `api_manager.py`

## 📈 Performance & Scaling

### Optimization Features

- **🚀 Parallel Processing**: Multi-threaded frame analysis
- **⏱️ Intelligent Rate Limiting**: Adaptive API call management
- **💾 Background Tasks**: Non-blocking analysis processing
- **🔄 Caching**: Efficient resource utilization
- **📊 Cost Tracking**: Real-time API usage monitoring

### Performance Metrics

- **Processing Speed**: ~2-5 minutes for 10 frames (comprehensive)
- **API Costs**: ~$0.02-0.05 per video (depends on length/depth)
- **Memory Usage**: ~500MB-2GB (depends on video size)
- **Concurrent Users**: Supports multiple simultaneous analyses

## 🛠️ Troubleshooting

### Common Issues

#### API Key Issues
```bash
# Check if API keys are set
echo $FIREWORKS_API_KEY
echo $PINECONE_API_KEY

# Test API connectivity
python -c "from fireworks_client import FireworksClient; FireworksClient().test_connection()"
```

#### Rate Limiting
- The system includes intelligent rate limiting
- Automatic retries with exponential backoff
- Cost optimization through smart API usage

#### Memory Issues
- Reduce `max_frames` for large videos
- Lower `fps_extract` for longer content
- Use `basic` analysis depth for quick processing

#### Search Not Working
- Ensure Pinecone API key is valid
- Check vector database initialization
- Verify embeddings are being generated

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- 🤖 New AI agent personalities and templates
- 🔍 Enhanced search capabilities
- 🎨 UI/UX improvements
- 📊 Additional analysis features
- 🧪 Test coverage expansion
- 📚 Documentation improvements

## 📋 Roadmap

### Version 2.0 (Coming Soon)
- [ ] 🎞️ Batch video processing
- [ ] 🌐 Multi-language support
- [ ] 📱 Mobile-responsive interface
- [ ] 🔄 Real-time video streaming analysis
- [ ] 📊 Advanced analytics dashboard
- [ ] 🔗 Webhook integrations
- [ ] 💾 Cloud storage integration

### Version 2.1 (Future)
- [ ] 🧠 Custom AI model training
- [ ] 🎭 Advanced agent behavior modification
- [ ] 📈 Enterprise features
- [ ] 🔐 Advanced security features
- [ ] 🌍 Multi-region deployment
- [ ] 📱 Mobile app

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Fireworks.ai** for providing powerful AI model APIs
- **Pinecone** for vector database services
- **OpenCV** community for video processing capabilities
- **FastAPI** team for the excellent web framework
- **Sentence Transformers** for embedding generation
- All contributors who help improve this project


## 🔗 Links

- [Medium](https://medium.com/@sreenikethaathreya/hollywoo-ai-video-analysis-5eae37ca7e16)
- [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7370126137122897920/)
- [Video Tutorials](https://www.youtube.com/watch?v=KBs_3V0OAgU&ab_channel=SreenikethAathreya)
- 📧 **Email**: sreenikethaathreya@gmail.com
---

<div align="center">

⭐ **Star this repo if you find it useful!** ⭐

[🚀 Get Started](#-quick-start) | [📖 Documentation](docs/) | [🤝 Contribute](#-contributing) | [💬 Support](#-support)

</div>