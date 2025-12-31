# Virtual Environment Setup Guide

This guide will help you set up a virtual environment for the NBA Similarity project.

## Why Use a Virtual Environment?

Virtual environments isolate your project dependencies from your system Python, preventing conflicts and keeping your project clean.

## Quick Setup

### Windows

**Option 1: Use the setup script**
```bash
setup_env.bat
```

**Option 2: Manual setup**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/Mac

**Option 1: Use the setup script**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

**Option 2: Manual setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Activating the Virtual Environment

### Windows
```bash
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Linux/Mac
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

## Deactivating the Virtual Environment

Simply run:
```bash
deactivate
```

## Using the Project

Once your virtual environment is activated:

1. **Preprocess your data:**
   ```bash
   python -m nba_similarity.cli preprocess --csv-file nbaplayersdraft.csv
   ```

2. **Train the models:**
   ```bash
   python -m nba_similarity.cli train --n-clusters 8
   ```

3. **Run the Streamlit app:**
   ```bash
   python -m nba_similarity.cli app
   ```

## Troubleshooting

### "python: command not found"
- On Linux/Mac, try `python3` instead of `python`
- Make sure Python is installed and in your PATH

### "venv: command not found"
- Make sure you're using Python 3.6 or higher
- Try `python3 -m venv venv` instead

### "pip: command not found"
- Make sure pip is installed: `python -m ensurepip --upgrade`
- Or install pip separately

### Virtual environment not activating
- Make sure you're in the project root directory
- Check that the `venv` folder was created successfully
- On Windows, try: `.\venv\Scripts\activate`

## Adding to .gitignore

The `venv` folder is already in `.gitignore`, so it won't be committed to version control.

## Best Practices

1. **Always activate** the virtual environment before working on the project
2. **Install new packages** while the venv is activated: `pip install package_name`
3. **Update requirements.txt** after installing new packages: `pip freeze > requirements.txt`
4. **Don't commit** the venv folder (already in .gitignore)

## Verifying Installation

After setup, verify everything works:

```bash
# Activate venv (if not already active)
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Check Python version
python --version

# Check installed packages
pip list

# Test the project
python -c "import nba_similarity; print('Success!')"
```

## Next Steps

Once your virtual environment is set up and activated:

1. Follow the [QUICKSTART.md](QUICKSTART.md) guide
2. Process your data
3. Train the models
4. Launch the Streamlit app

Happy coding! 🏀

