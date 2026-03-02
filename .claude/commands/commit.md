Summarize all uncommitted changes and create a git commit.

## Steps

1. Run `git status` and `git diff` to review all staged and unstaged changes
2. Run `git log --oneline -5` to follow the existing commit message style
3. Analyze the changes and write a concise commit message:
   - Subject line: short imperative summary (e.g. "add X", "fix Y", "refactor Z")
   - Body: bullet points covering what changed and why (group by area: backend, frontend, etc.)
   - End with: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
4. Stage relevant modified files (avoid secrets or unrelated files)
5. Commit using a HEREDOC to preserve formatting
6. Run `git status` to confirm success
7. Run `git push` to push the commit to the remote
