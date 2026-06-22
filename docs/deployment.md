# Deployment

The public documentation site is built as static MkDocs output and deployed via Cloudflare Workers static assets.

## Production deployment model

- Worker deployment style: Cloudflare Workers static assets
- Static output directory: `site/`
- Worker config file: `wrangler.jsonc`
- Production branch: `main`

## Cloudflare build settings

```text
Build command:
python -m pip install --upgrade pip && python -m pip install -r docs/requirements.txt && python -m mkdocs build --strict

Deploy command:
npx wrangler@4 deploy

Environment variable:
PYTHON_VERSION=3.12
```

Configure the Worker name and route/domain in `wrangler.jsonc` for your environment.

## Local docs build

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve -a 127.0.0.1:8000
```

## PyPI package deployment

Python package releases are deployed by `.github/workflows/publish.yml`. The
workflow builds the source distribution and wheel, checks the package metadata
with `twine`, installs the built wheel in a fresh environment, and publishes
the artifacts to PyPI.

The workflow uses PyPI Trusted Publishing with GitHub OIDC. Do not add a PyPI
password or API token to the repository.

Configure the PyPI project trusted publisher with these values:

```text
Publisher: GitHub
Owner: BlankenbergLab
Repository: galaxy-toolsmith
Workflow: publish.yml
Environment: pypi
```

Configure a GitHub environment named `pypi`. Use required reviewers on that
environment when releases should have a manual approval gate before upload.

Release tags must match the package version in `pyproject.toml` using the
`v<version>` form. For example, package version `0.1.0` must be released from
tag `v0.1.0`. PyPI does not allow replacing an already uploaded version, so
bump the package version before each public release.

Manual workflow dispatch is available for build-only verification by leaving
`publish` set to `false`. Setting `publish` to `true` uses the same `pypi`
environment and Trusted Publishing path as a GitHub release.
