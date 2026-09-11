# Workflow Templates

This directory contains reusable workflow templates aligned to framework/pattern usage.

## npm publish template

- File: `npm-publish.yml`
- Triggered via `workflow_call`
- Intended caller workflow: `.github/workflows/publish-npm.yml`

### Required repository secrets

- `NPM_TOKEN`: npm automation token with publish access to the package scope.

### Behavior

On merge to `main` (from caller workflow), it:

1. Installs dependencies (`npm ci`)
2. Runs tests/build if present
3. Checks whether `package.json` version already exists on npm
4. Publishes only when version is new
