# Contributing to SmartFace

## Git Workflow

### 1. Update your local copy
```bash
git pull origin main
```

### 2. Create a branch for your feature
```bash
# For Person 1 (ML):
git checkout -b feature/face-recognition

# For Person 2 (Backend):
git checkout -b feature/api-endpoints

# For Person 3 (Frontend):
git checkout -b feature/camera-feed
```

### 3. Make changes and commit
```bash
# Check what changed
git status

# Add your changes
git add .

# Commit with clear message
git commit -m "feat: Add real-time face recognition"
```

### Commit Message Format:
### 4. Push to GitHub
```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request
- Go to GitHub
- Click "Pull Requests" tab
- Click "New Pull Request"
- Select your branch
- Add description of changes
- Click "Create Pull Request"

### 6. Code Review
- Wait for teammates to review
- Make requested changes if needed
- Once approved, merge to main

### 7. Merge and Clean Up
```bash
# After merging on GitHub:
git checkout main
git pull origin main
git branch -d feature/your-feature-name
```

## Branch Naming Convention
## Commit Frequency
- ✅ Commit at least once per day
- ✅ Commit when a feature is done
- ✅ Commit after fixing a bug
- ❌ Don't commit massive changes all at once

## Collaboration Rules
1. Always pull before starting work
2. Never push directly to main
3. Create a Pull Request for review
4. Get approval before merging
5. Delete branch after merging

## Push/Pull Checklist
Before every git push:
- [ ] `git status` - check what's changing
- [ ] `git diff` - review your changes
- [ ] Tested locally
- [ ] No debugging code left
- [ ] Commit messages are clear

---

**Let's build something great! 🚀**
