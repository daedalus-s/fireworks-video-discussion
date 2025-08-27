"""
Complete Configurable Multi-Agent Discussion System - FIXED VERSION
Includes robust rate limiting to prevent 429 errors
"""

import os
import asyncio
import json
import logging
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from fireworks_client import FireworksClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced API Manager with Conservative Rate Limiting
class ImprovedAPIManager:
    """Enhanced API manager with conservative rate limiting for Fireworks AI"""
    
    def __init__(self):
        self.last_call_time = 0
        self.min_delay = 5.0  # 5 seconds minimum between calls
        self.consecutive_failures = 0
        self.backoff_multiplier = 1.0
        self.call_count = 0
        
    async def acquire(self):
        """Conservative rate limiting with dynamic backoff"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        # Calculate dynamic delay based on recent failures
        dynamic_delay = self.min_delay * self.backoff_multiplier
        
        # Add extra delay every 10 calls to prevent sustained high rate
        if self.call_count % 10 == 0 and self.call_count > 0:
            dynamic_delay *= 1.5
            logger.info(f"Batch delay: Extended wait after {self.call_count} calls")
        
        if time_since_last < dynamic_delay:
            wait_time = dynamic_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    async def call_with_retry(self, func, max_retries: int = 3, **kwargs):
        """Enhanced retry with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                await self.acquire()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)
                
                # Success - reduce backoff gradually
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
                self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.95)
                
                return result
                
            except Exception as e:
                if '429' in str(e):
                    self.consecutive_failures += 1
                    self.backoff_multiplier = min(3.0, 1.2 ** self.consecutive_failures)
                    
                    if attempt < max_retries - 1:
                        # Aggressive exponential backoff for rate limits
                        wait_time = (3 ** attempt) * 5 + random.uniform(2, 8)
                        logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts due to rate limiting")
                        raise e
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    else:
                        raise e
        
        raise Exception(f"Failed after {max_retries} attempts")

@dataclass
class CustomAgent:
    """Represents a custom AI agent with user-defined characteristics"""
    name: str
    role: str
    personality: str
    expertise: List[str]
    discussion_style: str
    model: str  # Which Fireworks model to use
    emoji: str
    focus_areas: List[str]
    analysis_approach: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomAgent':
        """Create agent from dictionary"""
        return cls(**data)

@dataclass
class AgentDiscussionTurn:
    """A single turn in the agent discussion"""
    agent_name: str
    agent_role: str
    content: str
    timestamp: str
    round_number: int
    responding_to: Optional[str] = None
    frame_references: List[Dict[str, Any]] = None
    timestamp_references: List[Dict[str, Any]] = None

class AgentTemplates:
    """Predefined agent templates for different content types"""
    
    @staticmethod
    def get_film_analysis_agents() -> List[CustomAgent]:
        """Film analysis specialist agents"""
        return [
            CustomAgent(
                name="Cinematographer Chen",
                role="Director of Photography",
                personality="Visually focused and technically precise. Analyzes camera work, lighting, and visual composition with professional expertise.",
                expertise=["cinematography", "camera movements", "lighting design", "visual composition", "color grading", "lens selection"],
                discussion_style="Technical and precise, references specific cinematographic techniques and equipment",
                model="vision",  # Best for visual analysis
                emoji="🎥",
                focus_areas=["camera work", "lighting", "visual composition", "cinematographic techniques"],
                analysis_approach="Technical analysis of visual elements, camera techniques, and photographic principles"
            ),
            CustomAgent(
                name="Isabella Film Critic",
                role="Film Studies Scholar",
                personality="Analytical and culturally aware. Evaluates films within broader cinematic contexts and artistic traditions.",
                expertise=["film theory", "cinematic history", "genre analysis", "narrative structure", "cultural critique", "artistic merit"],
                discussion_style="Scholarly and contextual, references film history and theoretical frameworks",
                model="qwen3",  # Great for creative analysis
                emoji="🎭",
                focus_areas=["narrative structure", "artistic merit", "cinematic tradition", "cultural context"],
                analysis_approach="Academic and theoretical, placing content within broader cinematic and cultural contexts"
            ),
            CustomAgent(
                name="Marcus Sound Designer",
                role="Audio Post-Production Specialist",
                personality="Aurally focused and detail-oriented. Analyzes sound design, music, and audio storytelling elements.",
                expertise=["sound design", "musical score", "dialogue editing", "ambient sound", "audio mixing", "acoustic storytelling"],
                discussion_style="Audio-focused and technical, discusses sound's role in storytelling",
                model="gpt_oss",  # Balanced analysis
                emoji="🎵",
                focus_areas=["sound design", "musical elements", "audio storytelling", "acoustic atmosphere"],
                analysis_approach="Audio-centric analysis focusing on sound's contribution to narrative and emotional impact"
            )
        ]
    
    @staticmethod
    def get_educational_agents() -> List[CustomAgent]:
        """Educational content specialist agents"""
        return [
            CustomAgent(
                name="Dr. Learning Specialist",
                role="Educational Psychology Expert",
                personality="Student-focused and pedagogically minded. Evaluates content for learning effectiveness and engagement.",
                expertise=["learning theory", "cognitive psychology", "educational design", "student engagement", "knowledge retention", "instructional methods"],
                discussion_style="Research-based and learner-centered, focuses on educational outcomes",
                model="qwen3",
                emoji="🧠",
                focus_areas=["learning effectiveness", "student engagement", "knowledge retention", "cognitive load"],
                analysis_approach="Evidence-based educational analysis focusing on learning outcomes and student experience"
            ),
            CustomAgent(
                name="Prof. Subject Expert",
                role="Domain Knowledge Specialist",
                personality="Content-focused and accuracy-driven. Ensures subject matter accuracy and depth.",
                expertise=["subject matter expertise", "content accuracy", "factual verification", "domain knowledge", "academic rigor", "information quality"],
                discussion_style="Authoritative and precise, emphasizes accuracy and completeness",
                model="gpt_oss",
                emoji="📚",
                focus_areas=["content accuracy", "subject depth", "factual correctness", "academic quality"],
                analysis_approach="Subject matter validation with emphasis on accuracy, completeness, and academic standards"
            )
        ]
    
    @staticmethod
    def get_marketing_agents() -> List[CustomAgent]:
        """Marketing content specialist agents"""
        return [
            CustomAgent(
                name="Alex Brand Strategist",
                role="Brand Strategy Director",
                personality="Brand-focused and strategic. Analyzes content for brand alignment and strategic messaging.",
                expertise=["brand strategy", "messaging consistency", "brand identity", "market positioning", "brand storytelling", "strategic communication"],
                discussion_style="Strategic and brand-focused, evaluates alignment with brand objectives",
                model="qwen3",
                emoji="🏢",
                focus_areas=["brand consistency", "strategic messaging", "brand identity", "market positioning"],
                analysis_approach="Strategic brand analysis focusing on message alignment and brand equity building"
            ),
            CustomAgent(
                name="Casey Conversion Specialist",
                role="Performance Marketing Expert",
                personality="Results-driven and data-focused. Evaluates content for conversion potential and performance metrics.",
                expertise=["conversion optimization", "performance metrics", "user journey", "call-to-action design", "funnel analysis", "behavioral psychology"],
                discussion_style="Data-driven and results-oriented, focuses on measurable outcomes",
                model="gpt_oss",
                emoji="📈",
                focus_areas=["conversion potential", "user journey", "performance metrics", "behavioral triggers"],
                analysis_approach="Performance-focused analysis emphasizing conversion optimization and measurable business outcomes"
            )
        ]

# Template registry
AGENT_TEMPLATES = {
    "film_analysis": AgentTemplates.get_film_analysis_agents,
    "educational": AgentTemplates.get_educational_agents,
    "marketing": AgentTemplates.get_marketing_agents
}

def load_agent_template(template_name: str) -> Optional[List[CustomAgent]]:
    """Load a predefined agent template"""
    if template_name in AGENT_TEMPLATES:
        try:
            agents = AGENT_TEMPLATES[template_name]()
            logger.info(f"✅ Loaded {template_name} template with {len(agents)} agents")
            return agents
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return None
    else:
        logger.warning(f"Template {template_name} not found")
        return None

def show_available_templates():
    """Display available agent templates"""
    print("\n📋 AVAILABLE AGENT TEMPLATES")
    print("="*50)
    
    template_descriptions = {
        "film_analysis": {
            "description": "Film & Cinema Analysis Specialists",
            "agents": ["Cinematographer Chen", "Isabella Film Critic", "Marcus Sound Designer"],
            "use_case": "Movie reviews, film analysis, cinematic content"
        },
        "educational": {
            "description": "Educational Content Specialists", 
            "agents": ["Dr. Learning Specialist", "Prof. Subject Expert"],
            "use_case": "Educational videos, tutorials, learning content"
        },
        "marketing": {
            "description": "Marketing & Brand Specialists",
            "agents": ["Alex Brand Strategist", "Casey Conversion Specialist"],
            "use_case": "Marketing videos, commercials, promotional content"
        }
    }
    
    for template_name, info in template_descriptions.items():
        print(f"\n🎯 {template_name}")
        print(f"   Description: {info['description']}")
        print(f"   Agents: {', '.join(info['agents'])}")
        print(f"   Best for: {info['use_case']}")

class ConfigurableMultiAgentDiscussion:
    """Multi-agent discussion system with configurable agents and fixed rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the configurable multi-agent system"""
        self.client = FireworksClient(api_key)
        self.agents: List[CustomAgent] = []
        self.discussion_history: List[AgentDiscussionTurn] = []
        self.agent_configs_file = "agent_configurations.json"
        
        # Use the improved API manager with conservative rate limiting
        self.api_manager = ImprovedAPIManager()
        
        # Load saved agent configurations
        self._load_agent_configurations()
        
        logger.info("✅ Configurable Multi-Agent Discussion System initialized")
    
    def create_default_agents(self) -> List[CustomAgent]:
        """Create default agent configurations"""
        default_agents = [
            CustomAgent(
                name="Alex",
                role="Technical Analyst",
                personality="Detail-oriented and analytical. Focuses on technical aspects with precision and expertise.",
                expertise=["cinematography", "camera techniques", "lighting", "editing", "visual effects", "technical production"],
                discussion_style="Precise and factual, supports arguments with specific technical observations",
                model="gpt_oss",
                emoji="🎬",
                focus_areas=["technical quality", "production values", "camera work", "visual composition"],
                analysis_approach="Analytical and methodical, examining technical execution and craftsmanship"
            ),
            CustomAgent(
                name="Maya",
                role="Creative Interpreter",
                personality="Imaginative and insightful. Explores artistic meanings, themes, and emotional impact.",
                expertise=["storytelling", "symbolism", "themes", "emotional impact", "cultural context", "artistic interpretation"],
                discussion_style="Thoughtful and empathetic, asks deep questions and explores creative meanings",
                model="qwen3",
                emoji="🎨",
                focus_areas=["artistic expression", "narrative themes", "emotional resonance", "symbolic content"],
                analysis_approach="Interpretive and contextual, seeking deeper meanings and artistic significance"
            ),
            CustomAgent(
                name="Jordan",
                role="Audience Advocate",
                personality="Practical and viewer-focused. Considers accessibility, engagement, and audience experience.",
                expertise=["user experience", "audience engagement", "accessibility", "clarity", "viewer psychology", "communication"],
                discussion_style="Direct and practical, focuses on real-world impact and viewer perspective",
                model="vision",
                emoji="👥",
                focus_areas=["audience engagement", "accessibility", "viewer experience", "communication effectiveness"],
                analysis_approach="User-centered and practical, evaluating content from the audience's perspective"
            ),
            CustomAgent(
                name="Affan",
                role="Financial Marketing Analyst",
                personality="Business-focused and strategic. Analyzes content for commercial viability and market impact.",
                expertise=["Finance", "Technical Finance", "Marketing", "ROI analysis", "market positioning", "commercial strategy"],
                discussion_style="Data-driven and strategic, focuses on business outcomes and financial metrics",
                model="gpt_oss",
                emoji="🤖",
                focus_areas=["Finance", "Technical Finance", "commercial viability", "market analysis"],
                analysis_approach="Business-centered analysis focusing on financial impact and market positioning"
            )
        ]
        return default_agents
    
    def add_agent(self, agent: CustomAgent) -> bool:
        """Add a new agent to the system"""
        try:
            # Check for duplicate names
            if any(existing.name.lower() == agent.name.lower() for existing in self.agents):
                logger.warning(f"Agent with name '{agent.name}' already exists")
                return False
            
            self.agents.append(agent)
            self._save_agent_configurations()
            logger.info(f"✅ Added agent: {agent.name} ({agent.role})")
            return True
        except Exception as e:
            logger.error(f"Failed to add agent: {e}")
            return False
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent by name"""
        try:
            original_count = len(self.agents)
            self.agents = [agent for agent in self.agents if agent.name.lower() != agent_name.lower()]
            
            if len(self.agents) < original_count:
                self._save_agent_configurations()
                logger.info(f"✅ Removed agent: {agent_name}")
                return True
            else:
                logger.warning(f"Agent '{agent_name}' not found")
                return False
        except Exception as e:
            logger.error(f"Failed to remove agent: {e}")
            return False
    
    def get_agent(self, agent_name: str) -> Optional[CustomAgent]:
        """Get an agent by name"""
        for agent in self.agents:
            if agent.name.lower() == agent_name.lower():
                return agent
        return None
    
    def list_agents(self) -> List[CustomAgent]:
        """Get list of all configured agents"""
        return self.agents.copy()
    
    async def conduct_discussion(self, 
                                video_analysis: Dict[str, Any],
                                num_rounds: int = 3,
                                selected_agents: Optional[List[str]] = None) -> List[AgentDiscussionTurn]:
        """
        Conduct multi-agent discussion with FIXED rate limiting
        
        Args:
            video_analysis: Results from video analysis
            num_rounds: Number of discussion rounds
            selected_agents: Specific agents to include in discussion
            
        Returns:
            List of discussion turns
        """
        
        if not self.agents:
            logger.error("No agents configured for discussion")
            return []
        
        # Select agents for discussion
        discussion_agents = self.agents
        if selected_agents:
            discussion_agents = [agent for agent in self.agents if agent.name in selected_agents]
        
        if not discussion_agents:
            logger.error("No valid agents selected for discussion")
            return []
        
        logger.info(f"Starting {num_rounds}-round discussion with {len(discussion_agents)} agents")
        logger.info("Using CONSERVATIVE rate limiting (5-15s delays) to prevent 429 errors")
        
        # Prepare video summary
        video_summary = self._prepare_video_summary(video_analysis)
        
        # Discussion topics based on agent expertise
        topics = self._generate_discussion_topics(discussion_agents, num_rounds)
        
        discussion_turns = []
        
        # Run discussion rounds with aggressive rate limiting
        for round_num in range(1, num_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_num}/{num_rounds}")
            logger.info(f"{'='*50}")
            
            # Select topic for this round
            topic = topics[min(round_num - 1, len(topics) - 1)]
            logger.info(f"Topic: {topic}")
            
            # Process agents sequentially with long delays
            for i, agent in enumerate(discussion_agents):
                logger.info(f"Processing agent {i+1}/{len(discussion_agents)}: {agent.name}")
                
                try:
                    # Generate agent response with retry and rate limiting
                    response = await self.api_manager.call_with_retry(
                        self._generate_agent_response,
                        agent=agent,
                        video_summary=video_summary,
                        current_topic=topic,
                        discussion_history=discussion_turns[-3:],  # Last 3 turns for context
                        round_number=round_num,
                        max_retries=3
                    )
                    
                    # Add to discussion history
                    turn = AgentDiscussionTurn(
                        agent_name=agent.name,
                        agent_role=agent.role,
                        content=response,
                        timestamp=datetime.now().isoformat(),
                        round_number=round_num,
                        responding_to=discussion_turns[-1].agent_name if discussion_turns else None
                    )
                    
                    discussion_turns.append(turn)
                    
                    # Log the turn
                    logger.info(f"\n{agent.emoji} {agent.name} ({agent.role}):")
                    logger.info(f"{response[:200]}...")
                    
                    # Long delay between agents in the same round (8-13 seconds)
                    if i < len(discussion_agents) - 1:
                        delay = 8 + random.uniform(2, 5)
                        logger.info(f"⏸️ Waiting {delay:.1f}s before next agent...")
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    logger.error(f"Agent {agent.name} failed: {e}")
                    # Create fallback turn
                    fallback_response = f"As a {agent.role}, I find this content noteworthy from the perspective of {', '.join(agent.focus_areas[:2])}. However, I'm experiencing some difficulty articulating my detailed analysis at the moment."
                    
                    turn = AgentDiscussionTurn(
                        agent_name=agent.name,
                        agent_role=agent.role,
                        content=fallback_response,
                        timestamp=datetime.now().isoformat(),
                        round_number=round_num
                    )
                    
                    discussion_turns.append(turn)
            
            # Extra long delay between rounds (15-25 seconds)
            if round_num < num_rounds:
                delay = 15 + random.uniform(5, 10)
                logger.info(f"\n🔄 Round {round_num} complete. Waiting {delay:.1f}s before round {round_num + 1}...")
                await asyncio.sleep(delay)
        
        self.discussion_history = discussion_turns
        logger.info(f"\n✅ Discussion complete with {len(discussion_turns)} turns")
        return discussion_turns
    
    def _generate_discussion_topics(self, agents: List[CustomAgent], num_rounds: int) -> List[str]:
        """Generate discussion topics based on agent expertise"""
        base_topics = [
            "Share your initial analysis and key observations about this video content.",
            "Discuss the most significant aspects from your area of expertise.",
            "Evaluate the effectiveness and quality of the content from your perspective.",
            "What improvements or notable strengths would you highlight?",
            "Provide your final assessment and any recommendations."
        ]
        
        # Customize topics based on agent expertise
        if len(agents) > 0:
            # Incorporate agent focus areas into topics
            agent_focuses = []
            for agent in agents:
                agent_focuses.extend(agent.focus_areas[:2])  # Take top 2 focus areas
            
            if agent_focuses:
                specialized_topics = [
                    f"Analyze the video considering {', '.join(agent_focuses[:3])} and other relevant aspects.",
                    f"Discuss how the content addresses {', '.join(agent_focuses[3:6])} and similar considerations.",
                    f"Evaluate the overall effectiveness in terms of {', '.join(agent_focuses[:2])} and quality."
                ]
                
                # Merge with base topics
                combined_topics = base_topics[:2] + specialized_topics + base_topics[3:]
                return combined_topics[:num_rounds]
        
        return base_topics[:num_rounds]
    
    async def _generate_agent_response(self,
                                      agent: CustomAgent,
                                      video_summary: str,
                                      current_topic: str,
                                      discussion_history: List[AgentDiscussionTurn],
                                      round_number: int) -> str:
        """Generate a response from a specific configured agent with error handling"""
        
        # Build context from recent discussion
        recent_discussion = "\n".join([
            f"{turn.agent_name}: {turn.content[:150]}..."
            for turn in discussion_history
        ]) if discussion_history else "This is the beginning of our discussion."
        
        # Create agent-specific prompt incorporating their configuration
        prompt = f"""You are {agent.name}, a {agent.role}.

AGENT PROFILE:
Personality: {agent.personality}
Expertise: {', '.join(agent.expertise)}
Focus Areas: {', '.join(agent.focus_areas)}
Discussion Style: {agent.discussion_style}
Analysis Approach: {agent.analysis_approach}

VIDEO CONTENT TO ANALYZE:
{video_summary}

RECENT DISCUSSION:
{recent_discussion}

CURRENT DISCUSSION TOPIC: {current_topic}

As {agent.name}, provide your perspective on this topic. Stay true to your personality, expertise, and analysis approach. 
Reference specific video content when possible. Keep your response focused and insightful (2-3 paragraphs).
If responding to others, acknowledge their points while adding your unique perspective from your area of expertise."""
        
        try:
            # Use the agent's designated model
            result = await self.client.analyze_text(
                text=prompt,
                model_type=agent.model,
                max_tokens=400
            )
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating response for {agent.name}: {e}")
            # Fallback response that maintains agent personality
            return f"As a {agent.role}, I find this content noteworthy from the perspective of {', '.join(agent.focus_areas[:2])}. However, I'm experiencing some difficulty articulating my detailed analysis at the moment. I'd appreciate hearing other perspectives to continue our discussion."
    
    def _prepare_video_summary(self, video_analysis: Dict[str, Any]) -> str:
        """Prepare video summary for agent discussion"""
        summary = f"""Video Analysis Summary:
- Duration: Approximately {video_analysis.get('frame_count', 0) * 5} seconds
- Frames analyzed: {video_analysis.get('frame_count', 0)}
- Subtitles: {'Yes' if video_analysis.get('subtitle_count', 0) > 0 else 'No'}

Key Visual Elements:
"""
        
        # Add frame analysis highlights
        if 'frame_analyses' in video_analysis:
            for i, frame in enumerate(video_analysis['frame_analyses'][:4]):  # Top 4 frames
                if frame.get('analysis') and not 'failed' in frame.get('analysis', '').lower():
                    summary += f"- Frame at {frame.get('timestamp', i*5):.1f}s: {frame.get('analysis', 'N/A')[:120]}...\n"
        
        # Add subtitle highlights
        if video_analysis.get('subtitle_count', 0) > 0 and 'subtitle_analyses' in video_analysis:
            summary += "\nKey Dialogue/Audio:\n"
            for sub in video_analysis['subtitle_analyses'][:2]:  # Top 2 subtitle segments
                if sub.get('analysis') and not 'failed' in sub.get('analysis', '').lower():
                    summary += f"- {sub.get('subtitle_range', 'N/A')}: {sub.get('analysis', 'N/A')[:100]}...\n"
        
        # Add overall analysis
        if 'overall_analysis' in video_analysis:
            summary += f"\nOverall Assessment:\n{video_analysis['overall_analysis'][:400]}..."
        
        return summary
    
    def _load_agent_configurations(self):
        """Load agent configurations from file"""
        try:
            if os.path.exists(self.agent_configs_file):
                with open(self.agent_configs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.agents = [CustomAgent.from_dict(agent_data) for agent_data in data.get('agents', [])]
                    logger.info(f"✅ Loaded {len(self.agents)} agent configurations")
            else:
                # Create default agents if no config file exists
                self.agents = self.create_default_agents()
                self._save_agent_configurations()
                logger.info("✅ Created default agent configurations")
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            # Fallback to defaults
            self.agents = self.create_default_agents()
    
    def _save_agent_configurations(self):
        """Save agent configurations to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "agents": [agent.to_dict() for agent in self.agents]
            }
            with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved {len(self.agents)} agent configurations")
        except Exception as e:
            logger.error(f"Failed to save agent configurations: {e}")
    
    def get_discussion_summary(self) -> Dict[str, Any]:
        """Get summary of the last discussion"""
        if not self.discussion_history:
            return {"error": "No discussion history available"}
        
        # Count contributions by agent
        agent_contributions = {}
        for turn in self.discussion_history:
            agent_contributions[turn.agent_name] = agent_contributions.get(turn.agent_name, 0) + 1
        
        # Get unique rounds
        rounds = list(set(turn.round_number for turn in self.discussion_history))
        
        return {
            "total_turns": len(self.discussion_history),
            "participating_agents": len(agent_contributions),
            "rounds_completed": len(rounds),
            "agent_contributions": agent_contributions,
            "discussion_duration": "N/A",  # Could calculate from timestamps
            "agents_configured": len(self.agents)
        }

# Test the fixed configurable system
async def test_fixed_configurable_agents():
    """Test the fixed configurable agent system"""
    print("="*60)
    print("FIXED CONFIGURABLE AGENT SYSTEM TEST")
    print("="*60)
    
    # Initialize system
    system = ConfigurableMultiAgentDiscussion()
    
    # Show current agents
    print(f"\nConfigured agents: {len(system.agents)}")
    for agent in system.agents:
        print(f"  {agent.emoji} {agent.name} ({agent.role}) - {agent.model}")
        print(f"    Expertise: {', '.join(agent.expertise[:3])}")
        print(f"    Focus: {', '.join(agent.focus_areas[:2])}")
    
    # Create mock video analysis for testing
    mock_analysis = {
        "video_path": "test_video.mp4",
        "frame_count": 3,
        "subtitle_count": 2,
        "frame_analyses": [
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "A cinematic establishing shot showing urban landscape with dramatic lighting."
            },
            {
                "frame_number": 2,
                "timestamp": 10.0,
                "analysis": "Close-up character portrait with emotional depth and professional cinematography."
            }
        ],
        "subtitle_analyses": [
            {
                "subtitle_range": "0.0s - 5.0s",
                "analysis": "Dialogue demonstrates strong character development and thematic resonance."
            }
        ],
        "overall_analysis": "Professional film content with strong cinematographic elements and compelling narrative structure."
    }
    
    # Test discussion with rate limiting
    print(f"\n💬 Testing fixed agent discussion with conservative rate limiting...")
    
    try:
        discussion = await system.conduct_discussion(
            video_analysis=mock_analysis,
            num_rounds=1,  # Test with just 1 round first
            selected_agents=["Alex", "Maya"]  # Test with just 2 agents
        )
        
        print(f"\n✅ Discussion complete with {len(discussion)} turns")
        for turn in discussion:
            agent = system.get_agent(turn.agent_name)
            emoji = agent.emoji if agent else "🤖"
            print(f"\n{emoji} {turn.agent_name}: {turn.content[:150]}...")
        
        # Test summary
        summary = system.get_discussion_summary()
        print(f"\n📊 Discussion Summary:")
        print(f"  Total turns: {summary['total_turns']}")
        print(f"  Participating agents: {summary['participating_agents']}")
        print(f"  Agent contributions: {summary['agent_contributions']}")
        
    except Exception as e:
        print(f"❌ Discussion test failed: {e}")
    
    print(f"\n✅ Fixed configurable agent system test complete!")

# CLI interface for agent configuration
def configure_agents_cli():
    """Command-line interface for agent configuration"""
    print("🤖 Multi-Agent Discussion System Configuration")
    print("=" * 60)
    
    system = ConfigurableMultiAgentDiscussion()
    
    while True:
        print(f"\nCurrent agents: {len(system.agents)}")
        for i, agent in enumerate(system.agents, 1):
            print(f"  {i}. {agent.emoji} {agent.name} ({agent.role})")
        
        print("\nConfiguration Options:")
        print("  1. View agent details")
        print("  2. Load agent template")
        print("  3. Add custom agent")
        print("  4. Remove agent")
        print("  5. Test discussion (1 round, 2 agents)")
        print("  6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            if system.agents:
                for agent in system.agents:
                    print(f"\n{agent.emoji} {agent.name} ({agent.role})")
                    print(f"  Model: {agent.model}")
                    print(f"  Expertise: {', '.join(agent.expertise[:4])}")
                    print(f"  Focus: {', '.join(agent.focus_areas[:3])}")
                    print(f"  Personality: {agent.personality[:100]}...")
            else:
                print("❌ No agents configured")
        
        elif choice == "2":
            show_available_templates()
            template_name = input("\nEnter template name to load: ").strip()
            template_agents = load_agent_template(template_name)
            if template_agents:
                replace = input("Replace all current agents? (y/N): ").strip().lower() == 'y'
                if replace:
                    system.agents = template_agents
                else:
                    existing_names = {agent.name.lower() for agent in system.agents}
                    new_agents = [agent for agent in template_agents if agent.name.lower() not in existing_names]
                    system.agents.extend(new_agents)
                system._save_agent_configurations()
                print(f"✅ Loaded template: {template_name}")
            else:
                print("❌ Failed to load template")
        
        elif choice == "3":
            print("Add custom agent functionality would go here")
            print("For now, use templates or edit the agent_configurations.json file directly")
        
        elif choice == "4":
            if system.agents:
                print("Available agents:")
                for i, agent in enumerate(system.agents, 1):
                    print(f"  {i}. {agent.name}")
                try:
                    choice_num = int(input("Select agent number to remove: "))
                    if 1 <= choice_num <= len(system.agents):
                        agent_to_remove = system.agents[choice_num - 1]
                        confirm = input(f"Remove {agent_to_remove.name}? (y/N): ").strip().lower()
                        if confirm == 'y':
                            system.remove_agent(agent_to_remove.name)
                        else:
                            print("❌ Removal cancelled")
                    else:
                        print("❌ Invalid selection")
                except ValueError:
                    print("❌ Invalid input")
            else:
                print("❌ No agents to remove")
        
        elif choice == "5":
            if len(system.agents) >= 2:
                print("Testing discussion with first 2 agents...")
                
                # Create simple test data
                test_data = {
                    "video_path": "test.mp4",
                    "frame_count": 2,
                    "subtitle_count": 1,
                    "frame_analyses": [{"analysis": "Test frame analysis"}],
                    "overall_analysis": "Test video for agent discussion"
                }
                
                async def run_test():
                    return await system.conduct_discussion(
                        video_analysis=test_data,
                        num_rounds=1,
                        selected_agents=[system.agents[0].name, system.agents[1].name]
                    )
                
                try:
                    import asyncio
                    discussion = asyncio.run(run_test())
                    print(f"✅ Test discussion completed with {len(discussion)} turns")
                except Exception as e:
                    print(f"❌ Test failed: {e}")
            else:
                print("❌ Need at least 2 agents for discussion test")
        
        elif choice == "6":
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "configure":
            configure_agents_cli()
        elif sys.argv[1] == "test":
            asyncio.run(test_fixed_configurable_agents())
        elif sys.argv[1] == "templates":
            show_available_templates()
        else:
            print("Usage: python configurable_agent_system.py [configure|test|templates]")
    else:
        print("Fixed Configurable Multi-Agent Discussion System")
        print("Usage:")
        print("  python configurable_agent_system.py configure  # Interactive configuration")
        print("  python configurable_agent_system.py test       # Test the system")
        print("  python configurable_agent_system.py templates  # Show available templates")