# Claude Code Commands for A2Z TSN Project

This file contains commonly used commands and configurations for the A2Z TSN/FRER project.

## Build and Test Commands

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Run Python tests
python -m pytest tests/ -v

# Run linting
python -m ruff check .
python -m ruff format .

# Type checking
python -m mypy config/ simulation/ ml/

# Run dashboard locally
python -m http.server 8000

# Docker build and run
docker-compose up -d

# CBS/FRER configuration generation
python config/cbs-frer/lan9662-cbs-config.py
python config/cbs-frer/lan9662-frer-config.py
```

## GitHub Pages Deployment

```bash
# Build and deploy to GitHub Pages
git add .
git commit -m "Update documentation and demos"
git push origin main

# Check GitHub Pages site
# https://hwkim3330.github.io/a2z/
```

## Development Workflow

1. **Documentation Updates**: Edit README.md or README_EN.md
2. **Code Changes**: Modify Python scripts in config/ or simulation/
3. **Web Updates**: Update HTML files in docs/
4. **Testing**: Run tests and validation scripts
5. **Commit**: Use descriptive commit messages
6. **Deploy**: Push to GitHub for automatic Pages deployment

## Quality Assurance Checklist

- [ ] All Python code passes ruff linting
- [ ] All links in documentation work
- [ ] GitHub Pages site renders correctly
- [ ] Interactive demos function properly
- [ ] Korean and English versions are synchronized
- [ ] All images and assets load correctly
- [ ] Performance metrics are accurate and up-to-date

## Project Structure

```
a2z/
├── config/cbs-frer/     # CBS/FRER configuration scripts
├── docs/                # GitHub Pages website
│   ├── api/            # API documentation
│   ├── ko/             # Korean language site
│   ├── simulation/     # Network simulator
│   └── topology/       # Network topology viewer
├── ml/                 # Machine learning components
├── simulation/         # Network simulation
├── assets/logo/        # Project logos and branding
└── README.md          # Main documentation
```

## Common Issues and Solutions

**Issue**: GitHub Pages not updating
**Solution**: Check _config.yml and ensure all HTML files are valid

**Issue**: Python import errors
**Solution**: Ensure all dependencies in requirements.txt are installed

**Issue**: Broken links in documentation
**Solution**: Use relative paths and test all links locally

**Issue**: High-DPI images appearing blurry
**Solution**: Use SVG format and proper CSS image-rendering properties