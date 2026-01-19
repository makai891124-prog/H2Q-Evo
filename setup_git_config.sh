#!/bin/bash
# H2Q-Evo Git Configuration Setup
# This script sets up git aliases for easy open source release

echo "🔧 Setting up Git aliases for H2Q-Evo..."

# Configure git aliases
git config --local alias.release '!bash -c "
echo \"📦 Preparing H2Q-Evo for release...\";
git add . && \
git commit -m \"feat: Open source release preparation\" && \
git push origin main && \
echo \"✅ Release preparation complete!\";
"'

git config --local alias.tag-release '!bash -c "
git tag -a v0.1.0 -m \"H2Q-Evo v0.1.0: Open Source Release\" && \
git push origin v0.1.0 && \
echo \"✅ Release tag created and pushed!\";
"'

git config --local alias.status-detailed "status --porcelain"

git config --local alias.last "log -1 --oneline"

# Display configured aliases
echo ""
echo "✅ Git aliases configured!"
echo ""
echo "Available aliases:"
echo "  git release       - Prepare and push release"
echo "  git tag-release   - Create and push version tag"
echo "  git status-detailed - Show detailed status"
echo "  git last          - Show last commit"
echo ""

# Show current configuration
echo "Current git configuration:"
git config --local --list | grep -E '(user|alias)' || true
