"""
Multi-Agent Discussion System
Different AI agents with distinct personalities discuss video content
"""

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random

from fireworks_client import FireworksClient
from video_analysis_system import VideoAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Agent:
    """Represents an AI agent with personality"""
    name: str
    role: str
    personality: str
    model: str  # Which Fireworks model to use
    expertise: List[str]
    discussion_style: str
    emoji: str

@dataclass
class DiscussionTurn:
    """A single turn in the discussion"""
    agent_name: str
    agent_role: str
    content: str
    timestamp: str
    responding_to: Optional[str] = None

class MultiAgentDiscussion:
    """Orchestrates multi-agent discussions about video content"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the multi-agent system"""
        self.client = FireworksClient(api_key)
        self.agents = self._create_agents()
        self.discussion_history: List[DiscussionTurn] = []
        
        logger.info("✅ Multi-Agent Discussion System initialized")
    
    def _create_agents(self) -> List[Agent]:
        """Create three agents with different personalities and models"""
        return [
            Agent(
                name="Alex",
                role="Technical Analyst",
                personality="Detail-oriented and analytical. Focuses on technical aspects, production quality, and visual composition.",
                model="gpt_oss",  # Uses GPT-OSS-120B
                expertise=["cinematography", "editing", "visual effects", "technical production"],
                discussion_style="Precise and factual, supports arguments with specific observations",
                emoji="🎬"
            ),
            Agent(
                name="Maya",
                role="Creative Interpreter",
                personality="Imaginative and insightful. Explores themes, emotions, and artistic meanings.",
                model="qwen3",  # Uses Qwen3-235B
                expertise=["storytelling", "symbolism", "emotional impact", "cultural context"],
                discussion_style="Thoughtful and empathetic, asks deep questions and explores meanings",
                emoji="🎨"
            ),
            Agent(
                name="Jordan",
                role="Audience Advocate",
                personality="Practical and viewer-focused. Considers accessibility, engagement, and audience experience.",
                model="vision",  # Uses Llama4 Maverick (also for text)
                expertise=["user experience", "engagement", "clarity", "audience psychology"],
                discussion_style="Direct and practical, focuses on real-world impact and viewer perspective",
                emoji="👥"
            )
        ]
    
    async def discuss_video(self, 
                           video_analysis: Dict[str, Any],
                           num_rounds: int = 3) -> List[DiscussionTurn]:
        """Generate a multi-agent discussion about the video
        
        Args:
            video_analysis: Results from video analysis
            num_rounds: Number of discussion rounds
            
        Returns:
            List of discussion turns
        """
        logger.info(f"Starting {num_rounds}-round discussion with {len(self.agents)} agents")
        
        # Prepare video summary for agents
        video_summary = self._prepare_video_summary(video_analysis)
        
        # Discussion topics
        topics = [
            "What are your initial impressions of this video?",
            "What stands out most from a technical or creative perspective?",
            "How effective is this video in achieving its apparent purpose?",
            "What improvements or observations would you suggest?",
            "What is the overall impact and memorability of this video?"
        ]
        
        # Run discussion rounds
        for round_num in range(num_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_num + 1}/{num_rounds}")
            logger.info(f"{'='*50}")
            
            # Select topic for this round
            topic = topics[min(round_num, len(topics)-1)]
            logger.info(f"Topic: {topic}")
            
            # Randomize agent order for natural discussion
            agents_order = self.agents.copy()
            if round_num > 0:
                random.shuffle(agents_order)
            
            for agent in agents_order:
                # Generate agent response
                response = await self._generate_agent_response(
                    agent=agent,
                    video_summary=video_summary,
                    current_topic=topic,
                    discussion_history=self.discussion_history[-3:]  # Last 3 turns for context
                )
                
                # Add to discussion history
                turn = DiscussionTurn(
                    agent_name=agent.name,
                    agent_role=agent.role,
                    content=response,
                    timestamp=datetime.now().isoformat(),
                    responding_to=self.discussion_history[-1].agent_name if self.discussion_history else None
                )
                
                self.discussion_history.append(turn)
                
                # Log the turn
                logger.info(f"\n{agent.emoji} {agent.name} ({agent.role}):")
                logger.info(f"{response[:200]}...")
                
                # Small delay for API rate limiting
                await asyncio.sleep(0.5)
        
        return self.discussion_history
    
    async def _generate_agent_response(self,
                                      agent: Agent,
                                      video_summary: str,
                                      current_topic: str,
                                      discussion_history: List[DiscussionTurn]) -> str:
        """Generate a response from a specific agent"""
        
        # Build context from recent discussion
        recent_discussion = "\n".join([
            f"{turn.agent_name}: {turn.content[:150]}..."
            for turn in discussion_history
        ]) if discussion_history else "This is the beginning of our discussion."
        
        # Create agent-specific prompt
        prompt = f"""You are {agent.name}, a {agent.role}.
        
Personality: {agent.personality}
Expertise: {', '.join(agent.expertise)}
Discussion style: {agent.discussion_style}

VIDEO SUMMARY:
{video_summary}

RECENT DISCUSSION:
{recent_discussion}

CURRENT TOPIC: {current_topic}

Provide your perspective as {agent.name}. Be specific and reference the video content. 
Keep your response concise (2-3 paragraphs max) but insightful.
If responding to others, acknowledge their points while adding your unique perspective."""
        
        try:
            # Use the agent's designated model
            if agent.model == "vision":
                # Llama4 Maverick can do text generation too
                result = await self.client.analyze_text(
                    text=prompt,
                    model_type="vision",  # This will use Llama4 Maverick
                    max_tokens=300
                )
            else:
                result = await self.client.analyze_text(
                    text=prompt,
                    model_type=agent.model,
                    max_tokens=300
                )
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating response for {agent.name}: {e}")
            # Fallback response
            return f"As a {agent.role}, I find this video interesting, though I'm having trouble articulating my specific thoughts at the moment."
    
    def _prepare_video_summary(self, video_analysis: Dict[str, Any]) -> str:
        """Prepare a summary of video analysis for agents"""
        summary = f"""Video Analysis Summary:
- Duration: Approximately {video_analysis.get('frame_count', 0) * 5} seconds
- Frames analyzed: {video_analysis.get('frame_count', 0)}
- Subtitles: {'Yes' if video_analysis.get('subtitle_count', 0) > 0 else 'No'}

Key Visual Elements:
"""
        
        # Add frame analysis highlights
        if 'frame_analyses' in video_analysis:
            for i, frame in enumerate(video_analysis['frame_analyses'][:3]):
                summary += f"- Frame at {frame.get('timestamp', i*5):.1f}s: {frame.get('analysis', 'N/A')[:100]}...\n"
        
        # Add subtitle highlights
        if video_analysis.get('subtitle_count', 0) > 0 and 'subtitle_analyses' in video_analysis:
            summary += "\nKey Dialogue/Audio:\n"
            for sub in video_analysis['subtitle_analyses'][:2]:
                summary += f"- {sub.get('subtitle_range', 'N/A')}: {sub.get('text_analyzed', 'N/A')[:100]}...\n"
        
        # Add overall analysis
        if 'overall_analysis' in video_analysis:
            summary += f"\nOverall Assessment:\n{video_analysis['overall_analysis'][:300]}..."
        
        return summary
    
    def save_discussion(self, output_path: str = "discussion_results.json"):
        """Save discussion to JSON file"""
        discussion_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": [asdict(agent) for agent in self.agents],
            "discussion": [asdict(turn) for turn in self.discussion_history],
            "total_turns": len(self.discussion_history),
            "usage": self.client.get_usage_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, indent=2)
        
        logger.info(f"✅ Discussion saved to {output_path}")
    
    def print_discussion(self):
        """Print the discussion in a readable format"""
        print("\n" + "="*60)
        print("MULTI-AGENT VIDEO DISCUSSION")
        print("="*60)
        
        for i, turn in enumerate(self.discussion_history, 1):
            agent = next((a for a in self.agents if a.name == turn.agent_name), None)
            emoji = agent.emoji if agent else "💬"
            
            print(f"\n{emoji} {turn.agent_name} ({turn.agent_role})")
            print("-"*40)
            print(turn.content)
            
            if turn.responding_to:
                print(f"\n[Responding to: {turn.responding_to}]")
        
        print("\n" + "="*60)
        print(f"Discussion complete: {len(self.discussion_history)} turns")
        usage = self.client.get_usage_summary()
        print(f"Total cost: ${usage['total_cost_usd']:.4f}")
        print("="*60)


# Test the multi-agent system
async def test_multi_agent_discussion():
    """Test the multi-agent discussion system"""
    print("="*60)
    print("MULTI-AGENT DISCUSSION TEST")
    print("="*60)
    
    # Create a sample video analysis result
    sample_analysis = {
        "video_path": "test_video.mp4",
        "frame_count": 3,
        "subtitle_count": 3,
        "frame_analyses": [
            {
                "frame_number": 0,
                "timestamp": 0.0,
                "analysis": "A red background with white text reading 'Test Frame 1' and 'Time: 0.0s'. Simple composition with centered text.",
                "tokens_used": 50,
                "cost": 0.01
            },
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "Similar frame with 'Test Frame 2' and different color scheme. Maintains consistent layout.",
                "tokens_used": 45,
                "cost": 0.01
            },
            {
                "frame_number": 2,
                "timestamp": 10.0,
                "analysis": "Final test frame showing 'Test Frame 3' with evolved color palette.",
                "tokens_used": 40,
                "cost": 0.01
            }
        ],
        "subtitle_analyses": [
            {
                "subtitle_range": "0.0s - 5.0s",
                "text_analyzed": "Welcome to this test video demonstration.",
                "analysis": "Opening greeting that sets an educational tone.",
                "tokens_used": 30,
                "cost": 0.005
            }
        ],
        "overall_analysis": "This appears to be a test video with simple frames showing text and color changes. It demonstrates basic video processing capabilities with clear frame markers and timing information. The minimalist design focuses on functionality over aesthetics.",
        "processing_time": 15.5,
        "total_cost": 0.035,
        "timestamp": datetime.now().isoformat()
    }
    
    # Initialize discussion system
    try:
        discussion_system = MultiAgentDiscussion()
    except Exception as e:
        print(f"❌ Failed to initialize discussion system: {e}")
        return
    
    # Run discussion
    try:
        print("\nStarting multi-agent discussion...")
        print("Agents participating:")
        for agent in discussion_system.agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role}) - Model: {agent.model}")
        
        print("\n" + "-"*40)
        
        # Generate discussion
        discussion = await discussion_system.discuss_video(
            video_analysis=sample_analysis,
            num_rounds=2  # Just 2 rounds for testing
        )
        
        # Print the discussion
        discussion_system.print_discussion()
        
        # Save results
        discussion_system.save_discussion("test_discussion.json")
        
        print("\n✅ Multi-agent discussion test complete!")
        print("📄 Full discussion saved to test_discussion.json")
        
    except Exception as e:
        print(f"❌ Discussion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_multi_agent_discussion())