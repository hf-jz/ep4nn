#!/usr/bin/env bash
# Publish ep4nn to GitHub Pages (docs/ folder). Needs `gh` authenticated.
# One command: bash deploy.sh      Override the target with EP4NN_REPO=owner/name
set -euo pipefail
cd "$(dirname "$0")"
REPO="${EP4NN_REPO:-hf-jz/ep4nn}"
USER_GH="$(gh api user --jq .login)"

git init -q -b main 2>/dev/null || true
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git -c user.name="${USER_GH}" -c user.email="${USER_GH}@users.noreply.github.com" \
      commit -q -m "ep4nn: growth contract, prerequisite scan, paper and site"
fi

if gh repo view "$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$REPO.git"
  git push -u origin HEAD:main
else
  gh repo create "$REPO" --public --source=. --remote=origin --push
fi

# Pages from the docs/ folder on main
gh api -X POST "repos/$REPO/pages" -f 'source[branch]=main' -f 'source[path]=/docs' 2>/dev/null \
  || gh api -X PUT "repos/$REPO/pages" -f 'source[branch]=main' -f 'source[path]=/docs' >/dev/null

gh api "repos/$REPO/pages" --jq '{status: .status, url: .html_url, branch: .source.branch, path: .source.path}'
echo "site: https://$(echo "$REPO" | cut -d/ -f1).github.io/$(echo "$REPO" | cut -d/ -f2)/"
