# Framework Patterns: Workflows

## Pattern: npm package publish on main merge

Use a lightweight caller workflow in `.github/workflows/` and place implementation details in `.github/workflow-templates/`.

### Caller

- Triggers on `push` to `main`
- Calls reusable workflow with `uses: ./.github/workflow-templates/npm-publish.yml`
- Uses `secrets: inherit`

### Template

- Trigger: `workflow_call`
- Uses Node 20 and npm registry URL
- Performs install/test/build
- Guards against duplicate publishes by checking npm for current version
- Publishes with provenance and `NODE_AUTH_TOKEN` from `NPM_TOKEN`
