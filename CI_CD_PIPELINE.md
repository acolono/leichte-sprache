# GitLab CI/CD Pipeline Documentation

## Overview

The GitLab CI/CD pipeline automatically builds and tests the Leichte Sprache package on every commit.

## Pipeline Stages

### 1. Test Stage

**Trigger:** All branches and merge requests
**Duration:** ~5-10 minutes (includes spaCy model download)

**What it does:**
- Sets up Python 3.11 environment
- Installs all dependencies
- Downloads German spaCy model (`de_core_news_lg`)
- Runs full test suite (`dev/run_all_tests.py`)
- Generates test reports

**Artifacts:**
- Test reports (available for 7 days)
- Located in `dev/test_reports/`

**Status:** Currently set to `allow_failure: true` to not block builds during test setup

### 2. Build Stage

**Trigger:** `main`, `master`, tags, and `release/*` branches
**Duration:** ~2-3 minutes

**What it does:**
- Uses lightweight Alpine Linux image
- Installs `uv` build tool
- Builds both wheel and source distribution
- Verifies package contents
- Saves build artifacts

**Artifacts:**
- `dist/leichte_sprache_rulez-0.1.0-py3-none-any.whl` (~6 MB)
- `dist/leichte_sprache_rulez-0.1.0.tar.gz` (~6 MB)
- Available for 30 days
- Downloadable from GitLab UI

### 3. Release Stage (Tags only)

**Trigger:** Git tags (e.g., `v0.1.0`)
**Duration:** < 1 minute

**What it does:**
- Creates GitLab Release
- Attaches build artifacts to release
- Generates release notes

## Usage

### Viewing Pipeline Status

```bash
# In GitLab UI:
# Project → CI/CD → Pipelines
```

### Downloading Artifacts

1. Go to **CI/CD → Pipelines**
2. Click on the pipeline you want
3. Click **Download** next to the build job
4. Extract the artifacts

### Creating a Release

```bash
# Tag a commit for release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Pipeline will automatically:
# 1. Run tests
# 2. Build package
# 3. Create GitLab release
```

## Configuration

### CI/CD Variables

Set these in **Settings → CI/CD → Variables**:

| Variable | Description | Required |
|----------|-------------|----------|
| `TWINE_USERNAME` | PyPI username | Only for PyPI deployment |
| `TWINE_PASSWORD` | PyPI password/token | Only for PyPI deployment |

### Caching

The pipeline caches:
- **pip packages** (`.cache/pip/`)
- **uv cache** (`.cache/uv/`)
- **Virtual environment** (`.venv/`)

Cache is shared per branch to speed up subsequent runs.

## Troubleshooting

### Test Stage Fails

**Problem:** Tests fail or timeout
**Solution:**
- Tests are currently allowed to fail (`allow_failure: true`)
- Check test reports in artifacts
- Adjust test configuration in `dev/run_all_tests.py`

### Build Stage Fails: "personalpronomen not found"

**Problem:** Local dependency path `../personalpronomen/` doesn't exist in CI
**Solution:**
- Option 1: Include `personalpronomen` wheel in repository
- Option 2: Use git submodule
- Option 3: Build from source in CI
- Option 4: Remove from dependencies if not needed

### Build Stage Fails: Unicode Error

**Problem:** README.md encoding issue
**Solution:** Ensure README.md is UTF-8 encoded (already fixed)

### spaCy Model Download Too Slow

**Problem:** `de_core_news_lg` download takes 5+ minutes
**Solutions:**
- Cache the model in `.cache/` directory
- Use custom Docker image with pre-installed model
- Download once and use as artifact

## Advanced Configuration

### Deploy to PyPI (Commented Out)

To enable automatic PyPI deployment:

1. Uncomment the `deploy:pypi` job in `.gitlab-ci.yml`
2. Set `TWINE_USERNAME` and `TWINE_PASSWORD` variables
3. Push a tag: `git push origin v0.1.0`
4. Manual approval required (when: manual)

### Deploy to GitLab Package Registry

```yaml
deploy:gitlab:
  stage: deploy
  image: python:3.11-slim
  script:
    - pip install twine
    - TWINE_PASSWORD=${CI_JOB_TOKEN} TWINE_USERNAME=gitlab-ci-token python -m twine upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi dist/*
  only:
    - tags
```

### Custom Branch Rules

Edit the `only:` sections in `.gitlab-ci.yml`:

```yaml
# Build on feature branches too
only:
  - main
  - master
  - tags
  - /^feature\/.*$/
  - /^release\/.*$/
```

### Parallel Test Execution

```yaml
test:
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.11", "3.12"]
  image: python:${PYTHON_VERSION}-slim
```

## Performance Tips

1. **Use Docker image with pre-installed dependencies:**
   ```dockerfile
   FROM python:3.11-slim
   RUN pip install uv && python -m spacy download de_core_news_lg
   ```

2. **Cache more aggressively:**
   ```yaml
   cache:
     key: ${CI_COMMIT_REF_SLUG}
     policy: pull-push  # Upload cache after each job
   ```

3. **Skip tests for documentation changes:**
   ```yaml
   test:
     except:
       changes:
         - "**/*.md"
         - "docs/**/*"
   ```

## Monitoring

### Pipeline Duration Breakdown

Typical pipeline times:
- **Test stage:** 8-12 minutes (first run), 3-5 minutes (cached)
- **Build stage:** 2-3 minutes (first run), 1-2 minutes (cached)
- **Total:** ~10-15 minutes

### Success Rates

Check pipeline success rates:
```bash
# GitLab UI: Analytics → CI/CD Analytics
```

## Next Steps

- [ ] Fix personalpronomen dependency for CI
- [ ] Enable test stage requirement (remove `allow_failure: true`)
- [ ] Add code quality checks (pylint, black, mypy)
- [ ] Set up automated deployment to PyPI
- [ ] Create Docker image with pre-installed dependencies
- [ ] Add security scanning (SAST, dependency scanning)

## Support

For pipeline issues:
1. Check job logs in GitLab UI
2. Review this documentation
3. Consult GitLab CI/CD docs: https://docs.gitlab.com/ee/ci/

## See Also

- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Python Package Publishing](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [uv Build Tool](https://github.com/astral-sh/uv)
