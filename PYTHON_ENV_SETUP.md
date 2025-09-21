# Python Environment Auto-Activation Setup

This document explains the global Python environment setup that automatically activates virtual environments across all your projects.

## 🎯 What's Been Configured

### 1. Global Cursor Settings
**Location**: `~/Library/Application Support/Cursor/User/settings.json`

Added global Python settings that work across ALL projects:
- `python.terminal.activateEnvironment: true` - Auto-activate venv in terminals
- `python.terminal.activateEnvInCurrentTerminal: true` - Activate in current terminal
- `python.venvFolders: [".venv", "venv", "env"]` - Search for these venv directories
- `python.defaultInterpreterPath: "./.venv/bin/python"` - Default to local .venv

### 2. Shell Integration with direnv
**Location**: `~/.bashrc` and `~/.bash_profile`

Added `eval "$(direnv hook bash)"` to automatically load environment configurations when entering directories.

### 3. Project-Specific Auto-Activation
**Location**: `.envrc` files in project directories

Each project with a `.envrc` file will automatically:
- Create `.venv` if it doesn't exist
- Activate the virtual environment
- Set project-specific environment variables
- Display activation status

## 🚀 How It Works

### For Cursor/VS Code:
1. **Open any Python project** → Cursor automatically detects `.venv`
2. **Open terminal** → Virtual environment auto-activates
3. **Python interpreter** → Automatically uses `.venv/bin/python`

### For Terminal:
1. **Enter project directory** → `direnv` automatically runs `.envrc`
2. **Virtual environment activates** → No manual `source .venv/bin/activate` needed
3. **Leave directory** → Environment automatically deactivates

## 📁 Template System

### Quick Setup for New Projects:
```bash
# Use the setup script
~/.config/direnv/setup-python-project.sh my-new-project

# Or manually copy template
cp ~/.config/direnv/templates/python-venv.envrc /path/to/project/.envrc
cd /path/to/project
direnv allow
```

### Template Locations:
- `~/.config/direnv/templates/python-venv.envrc` - Template .envrc file
- `~/.config/direnv/setup-python-project.sh` - Project setup script

## 🔧 Usage Examples

### Starting a New Python Project:
```bash
mkdir my-awesome-project
cd my-awesome-project
cp ~/.config/direnv/templates/python-venv.envrc .envrc
direnv allow
# Virtual environment is now automatically created and activated!
```

### Working with Existing Projects:
```bash
cd existing-python-project
# If .envrc exists and is allowed, environment auto-activates
# If not, create .envrc from template and run 'direnv allow'
```

### In Cursor:
1. Open any Python project folder
2. Open integrated terminal
3. Virtual environment is automatically active
4. Python interpreter automatically points to `.venv/bin/python`

## 🎉 Benefits

✅ **No more manual activation** - Environments activate automatically  
✅ **Consistent across projects** - Same setup works everywhere  
✅ **IDE integration** - Cursor automatically uses the right Python  
✅ **Terminal integration** - Shell automatically activates environments  
✅ **Template system** - Quick setup for new projects  
✅ **Clean isolation** - Each project has its own environment  

## 🛠️ Troubleshooting

### If direnv doesn't work:
```bash
# Reload your shell configuration
source ~/.bashrc
# Or restart your terminal
```

### If .envrc isn't loading:
```bash
cd your-project
direnv allow
```

### If virtual environment isn't created:
```bash
# Check Python is available
python3 --version
# Manually create if needed
python3 -m venv .venv
```

### Reset everything:
```bash
# Remove project venv
rm -rf .venv
# Re-enter directory to trigger recreation
cd . && direnv reload
```

## 📝 Files Modified/Created

### Global Configuration:
- `~/Library/Application Support/Cursor/User/settings.json` - Global Cursor settings
- `~/.bashrc` - Shell direnv integration
- `~/.bash_profile` - macOS terminal compatibility

### Template System:
- `~/.config/direnv/templates/python-venv.envrc` - Template for new projects
- `~/.config/direnv/setup-python-project.sh` - Project setup script

### Project-Specific:
- `.envrc` - Auto-activation configuration (per project)
- `.venv/` - Virtual environment (auto-created)

This setup ensures that every Python project you work on will automatically have its virtual environment activated in both Cursor and your terminal, without any manual intervention! 🎯


