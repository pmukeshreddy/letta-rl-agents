"""
Skill Loader

Loads and manages skills from various sources:
- File system (.md files)
- Database
- Remote URLs
"""

from pathlib import Path
from typing import Optional
import re
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text.split()) * 1.3


class SkillParser:
    """
    Parse .md skill files into structured format.
    
    Expected format:
    ```markdown
    # Skill Name
    
    Brief description on first paragraph.
    
    ## When to Use
    - condition 1
    - condition 2
    
    ## Approach
    Step by step instructions...
    
    ## Examples
    ```code
    example code
    ```
    
    ## Pitfalls
    Common mistakes to avoid...
    ```
    """
    
    def parse(self, content: str, file_path: Optional[str] = None) -> dict:
        """
        Parse skill markdown content.
        
        Returns:
            {
                "name": str,
                "description": str,
                "content": str (full content),
                "sections": dict,
                "token_count": int,
                "file_path": str or None,
            }
        """
        lines = content.strip().split("\n")
        
        # Extract name from first H1
        name = "Untitled Skill"
        for line in lines:
            if line.startswith("# "):
                name = line[2:].strip()
                break
        
        # Extract description (first paragraph after title)
        description = ""
        in_description = False
        for line in lines:
            if line.startswith("# "):
                in_description = True
                continue
            if in_description:
                if line.strip() == "":
                    if description:
                        break
                    continue
                if line.startswith("#"):
                    break
                description += line + " "
        
        description = description.strip()
        
        # Parse sections
        sections = self._parse_sections(content)
        
        # Count tokens
        token_count = count_tokens(content)
        
        return {
            "name": name,
            "description": description,
            "content": content,
            "sections": sections,
            "token_count": token_count,
            "file_path": file_path,
        }
    
    def _parse_sections(self, content: str) -> dict:
        """Extract sections by H2 headers."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                
                current_section = line[3:].strip().lower().replace(" ", "_")
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        return sections
    
    def extract_id(self, file_path: str) -> str:
        """Extract skill ID from file path."""
        path = Path(file_path)
        return path.stem.lower().replace(" ", "-").replace("_", "-")
    
    def infer_category(self, name: str, content: str) -> str:
        """Infer skill category from name and content."""
        text = (name + " " + content).lower()
        
        categories = {
            "file-ops": ["file", "pdf", "csv", "excel", "document", "read", "write", "parse"],
            "api": ["api", "http", "rest", "endpoint", "request", "response", "fetch"],
            "data": ["data", "json", "xml", "database", "sql", "query", "transform"],
            "coding": ["code", "python", "javascript", "function", "class", "debug"],
            "web": ["web", "html", "css", "scrape", "browser", "dom"],
            "ai": ["llm", "prompt", "embedding", "model", "ai", "ml"],
        }
        
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def extract_tags(self, content: str) -> list[str]:
        """Extract tags from content."""
        tags = []
        
        # Look for explicit tags
        tag_match = re.search(r"tags?:\s*\[([^\]]+)\]", content, re.IGNORECASE)
        if tag_match:
            tags = [t.strip().strip('"\'') for t in tag_match.group(1).split(",")]
        
        return tags


class SkillLoader:
    """
    Load skills from file system.
    """
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.parser = SkillParser()
    
    def load_skill(self, file_path: str) -> dict:
        """Load single skill from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {file_path}")
        
        content = path.read_text(encoding="utf-8")
        skill = self.parser.parse(content, str(path))
        
        # Add derived fields
        skill["id"] = self.parser.extract_id(str(path))
        skill["category"] = self.parser.infer_category(skill["name"], content)
        skill["tags"] = self.parser.extract_tags(content)
        
        return skill
    
    def load_all(self) -> list[dict]:
        """Load all skills from directory."""
        skills = []
        
        if not self.skills_dir.exists():
            return skills
        
        for path in self.skills_dir.glob("**/*.md"):
            try:
                skill = self.load_skill(str(path))
                skills.append(skill)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
        
        return skills
    
    def load_by_category(self, category: str) -> list[dict]:
        """Load skills filtered by category."""
        all_skills = self.load_all()
        return [s for s in all_skills if s.get("category") == category]
    
    def watch(self, callback):
        """
        Watch for skill file changes.
        
        Calls callback(event_type, skill) on changes.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class Handler(FileSystemEventHandler):
                def __init__(self, loader, cb):
                    self.loader = loader
                    self.cb = cb
                
                def on_modified(self, event):
                    if event.src_path.endswith(".md"):
                        skill = self.loader.load_skill(event.src_path)
                        self.cb("modified", skill)
                
                def on_created(self, event):
                    if event.src_path.endswith(".md"):
                        skill = self.loader.load_skill(event.src_path)
                        self.cb("created", skill)
                
                def on_deleted(self, event):
                    if event.src_path.endswith(".md"):
                        skill_id = self.loader.parser.extract_id(event.src_path)
                        self.cb("deleted", {"id": skill_id})
            
            observer = Observer()
            observer.schedule(Handler(self, callback), str(self.skills_dir), recursive=True)
            observer.start()
            return observer
            
        except ImportError:
            print("watchdog not installed. File watching disabled.")
            return None


class SkillManager:
    """
    High-level skill management.
    
    Combines:
    - File loading
    - Database sync
    - Embedding computation
    """
    
    def __init__(
        self,
        loader: SkillLoader,
        repository: "SkillRepository",  # From db/models.py
        embedder: "SkillEmbedder",  # From selector/embeddings.py
    ):
        self.loader = loader
        self.repository = repository
        self.embedder = embedder
    
    def sync_from_files(self) -> dict:
        """
        Sync skills from files to database.
        
        Returns:
            {"created": int, "updated": int, "errors": int}
        """
        from ..db.models import Skill as SkillModel
        
        stats = {"created": 0, "updated": 0, "errors": 0}
        
        for skill_data in self.loader.load_all():
            try:
                # Check if exists
                existing = self.repository.get(skill_data["id"])
                
                # Compute embedding
                embedding = self.embedder.embed_skill(
                    skill_data["name"],
                    skill_data["content"],
                )
                embedding_bytes = embedding.tobytes()
                
                if existing:
                    # Update
                    existing.name = skill_data["name"]
                    existing.description = skill_data["description"]
                    existing.content = skill_data["content"]
                    existing.category = skill_data["category"]
                    existing.tags = skill_data["tags"]
                    existing.token_count = skill_data["token_count"]
                    existing.file_path = skill_data["file_path"]
                    existing.embedding = embedding_bytes
                    stats["updated"] += 1
                else:
                    # Create
                    skill = SkillModel(
                        id=skill_data["id"],
                        name=skill_data["name"],
                        description=skill_data["description"],
                        content=skill_data["content"],
                        category=skill_data["category"],
                        tags=skill_data["tags"],
                        token_count=skill_data["token_count"],
                        file_path=skill_data["file_path"],
                        embedding=embedding_bytes,
                    )
                    self.repository.create(skill)
                    stats["created"] += 1
                    
            except Exception as e:
                print(f"Error syncing {skill_data.get('id', 'unknown')}: {e}")
                stats["errors"] += 1
        
        return stats
    
    def get_skill_content(self, skill_id: str) -> Optional[str]:
        """Get skill content by ID."""
        skill = self.repository.get(skill_id)
        if skill:
            return skill.content
        return None
    
    def get_skills_for_injection(self, skill_ids: list[str]) -> list[dict]:
        """
        Get skills formatted for context injection.
        
        Returns list of {"id", "name", "content"} dicts.
        """
        skills = []
        for skill_id in skill_ids:
            skill = self.repository.get(skill_id)
            if skill:
                skills.append({
                    "id": skill.id,
                    "name": skill.name,
                    "content": skill.content,
                })
        return skills
