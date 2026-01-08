"""
Letta Client

Wrapper for Letta API to execute tasks with skill injection.
"""

from typing import Optional
import os
import json


class LettaClient:
    """
    Client for Letta agent execution.
    
    Handles:
    - Agent creation/management
    - Skill injection into context
    - Task execution
    - Response parsing
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.letta.ai",
        default_model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("LETTA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        
        self._client = None
        self._agents: dict[str, str] = {}  # name -> id
    
    def _get_client(self):
        """Lazy init Letta client."""
        if self._client is None:
            try:
                from letta import Letta
                self._client = Letta(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("letta package not installed. Run: pip install letta")
        return self._client
    
    def create_agent(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Create a new Letta agent.
        
        Returns agent_id.
        """
        client = self._get_client()
        
        agent = client.agents.create(
            name=name,
            model=model or self.default_model,
            system_prompt=system_prompt or "You are a helpful AI assistant.",
        )
        
        self._agents[name] = agent.id
        return agent.id
    
    def get_or_create_agent(
        self,
        name: str = "skill-executor",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Get existing agent or create new one."""
        if name in self._agents:
            return self._agents[name]
        
        client = self._get_client()
        
        # Try to find existing
        agents = client.agents.list()
        for agent in agents:
            if agent.name == name:
                self._agents[name] = agent.id
                return agent.id
        
        # Create new
        return self.create_agent(name, system_prompt)
    
    def _format_skills_prompt(self, skills: list[dict]) -> str:
        """
        Format skills as system prompt injection.
        
        Skills are injected as tagged blocks that the LLM can reference.
        """
        if not skills:
            return ""
        
        parts = ["\n\n## Available Skills\n"]
        parts.append("Use the following skills to help complete the task:\n")
        
        for skill in skills:
            parts.append(f"\n<skill id=\"{skill['id']}\" name=\"{skill['name']}\">")
            parts.append(skill["content"])
            parts.append("</skill>\n")
        
        return "".join(parts)
    
    def execute_with_skills(
        self,
        task: str,
        skills: list[dict],
        agent_id: Optional[str] = None,
        timeout: int = 60,
    ) -> dict:
        """
        Execute task with skills injected into context.
        
        Args:
            task: Task description / user message
            skills: List of {"id", "name", "content"} dicts
            agent_id: Specific agent to use (creates default if None)
            timeout: Request timeout in seconds
            
        Returns:
            {
                "success": bool,
                "response": str,
                "error": str or None,
                "tokens_used": int,
                "messages": list,
            }
        """
        client = self._get_client()
        
        # Get agent
        if agent_id is None:
            agent_id = self.get_or_create_agent()
        
        # Build message with skills context
        skills_context = self._format_skills_prompt(skills)
        
        if skills_context:
            full_message = f"{skills_context}\n\n## Task\n{task}"
        else:
            full_message = task
        
        try:
            # Send message to agent
            response = client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": full_message}],
            )
            
            # Parse response
            assistant_messages = [
                m for m in response.messages 
                if m.role == "assistant"
            ]
            
            response_text = ""
            if assistant_messages:
                response_text = assistant_messages[-1].content
            
            # Estimate tokens (rough)
            tokens_used = len(full_message.split()) + len(response_text.split())
            
            return {
                "success": True,
                "response": response_text,
                "error": None,
                "tokens_used": tokens_used * 2,  # Rough token estimate
                "messages": [{"role": m.role, "content": m.content} for m in response.messages],
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": str(e),
                "tokens_used": 0,
                "messages": [],
            }
    
    def execute_simple(
        self,
        task: str,
        agent_id: Optional[str] = None,
    ) -> dict:
        """Execute task without skill injection."""
        return self.execute_with_skills(task, skills=[], agent_id=agent_id)
    
    def delete_agent(self, agent_id: str):
        """Delete an agent."""
        client = self._get_client()
        client.agents.delete(agent_id)
        
        # Remove from cache
        self._agents = {k: v for k, v in self._agents.items() if v != agent_id}
    
    def list_agents(self) -> list[dict]:
        """List all agents."""
        client = self._get_client()
        agents = client.agents.list()
        return [{"id": a.id, "name": a.name} for a in agents]


class MockLettaClient:
    """
    Mock client for testing without Letta API.
    
    Simulates success/failure based on skill matching.
    """
    
    def __init__(self, success_rate: float = 0.7):
        self.success_rate = success_rate
        self._call_count = 0
    
    def get_or_create_agent(self, name: str = "mock", **kwargs) -> str:
        return "mock-agent-id"
    
    def execute_with_skills(
        self,
        task: str,
        skills: list[dict],
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Mock execution with simulated outcomes.
        
        Success probability increases with more relevant skills.
        """
        import random
        
        self._call_count += 1
        
        # Base success rate
        prob = self.success_rate
        
        # Bonus for having skills
        if skills:
            prob += 0.1 * min(len(skills), 3)
        
        # Simulate relevance check (keyword matching)
        task_lower = task.lower()
        relevant_skills = 0
        for skill in skills:
            skill_name = skill.get("name", "").lower()
            if any(word in task_lower for word in skill_name.split("-")):
                relevant_skills += 1
        
        prob += 0.05 * relevant_skills
        prob = min(prob, 0.95)  # Cap at 95%
        
        success = random.random() < prob
        
        if success:
            response = f"Task completed successfully using {len(skills)} skills."
        else:
            response = "I encountered an error while processing the task."
        
        return {
            "success": success,
            "response": response,
            "error": None if success else "Simulated failure",
            "tokens_used": 500 + len(skills) * 200,
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant", "content": response},
            ],
        }
    
    def execute_simple(self, task: str, **kwargs) -> dict:
        return self.execute_with_skills(task, skills=[], **kwargs)
