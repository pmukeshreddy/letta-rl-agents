# Git Operations

Version control with Git.

## When to Use
- Tracking code changes
- Collaborating with team
- Managing releases
- Recovering from mistakes

## Approach
1. Commit early and often
2. Write meaningful commit messages
3. Use branches for features
4. Review before merging

## Code Examples

```bash
# Basic workflow
git add .
git commit -m "feat: add user authentication"
git push origin main

# Branching
git checkout -b feature/new-feature
git push -u origin feature/new-feature

# Merging
git checkout main
git merge feature/new-feature

# Rebase for clean history
git rebase main

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Stash changes
git stash
git stash pop

# View history
git log --oneline --graph

# Find bug with bisect
git bisect start
git bisect bad HEAD
git bisect good v1.0.0
```

```python
# Using GitPython
from git import Repo

repo = Repo(".")
repo.index.add(["file.py"])
repo.index.commit("Update file")
origin = repo.remote("origin")
origin.push()
```

## Pitfalls
- Never force push to shared branches
- Don't commit secrets/credentials
- Large files belong in Git LFS
- Resolve conflicts carefully — test after
