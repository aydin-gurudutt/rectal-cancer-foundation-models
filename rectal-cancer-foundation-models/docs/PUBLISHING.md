# Publishing to GitHub (private, after lab authorization)

Two GitHub accounts are connected in ChatGPT (school and personal), but this project has no existing target repository and the available GitHub connector cannot create a new repository. **Nothing has been uploaded to GitHub yet.** Choose the owner with your supervisor and obtain permission to store research code and aggregate outputs on GitHub.

Using GitHub CLI (`gh`) on your own machine after approval:

```bash
cd rectal-cancer-foundation-models
python3 scripts/validate_results.py
git init -b main
git add README.md .gitignore LICENSE_NOTICE.md docs/ src/ scripts/ results/ figures/
git diff --cached --name-only  # examine for disallowed files/IDs
git commit -m "Organize Tasks 1–3 methods and aggregate results"
gh auth login
gh repo create rectal-cancer-foundation-models --private --source=. --remote=origin --push
```

Do not run `gh repo create` until the correct GitHub account is authenticated and your lab authorizes this storage. GitHub repository name and owner can be changed before publication.

The original custom Task 1–3 exported scripts are already staged under `src/`. Before publication, review the source for hardcoded filesystem paths, sensitive slide IDs, credentials, and dependencies. If CLAM/TRIDENT had lab modifications, preserve a reviewed patch or pin a private source reference rather than copying an unreviewed third-party tree. Do not blindly sync any output directory.
