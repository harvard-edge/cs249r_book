# ============================================================================
# TinyTorch Installer
# ============================================================================
#
# USAGE
# -----
#   iwr -useb mlsysbook.ai/tinytorch/install.ps1 | powershell -ExecutionPolicy Bypass
#
# ============================================================================

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================
$REPO_URL      = "https://github.com/harvard-edge/cs249r_book.git"
$REPO_SHORT    = "harvard-edge/cs249r_book"
$BRANCH        = "dev"
$INSTALL_DIR   = "tinytorch"
$SPARSE_PATH   = "tinytorch"
$TINYTORCH_VERSION = "0.1.0"

# ============================================================================
# ANSI Color Codes
# ============================================================================
$esc = [char]27
$RED   = "$esc[31m"
$GREEN = "$esc[32m"
$YELLOW= "$esc[33m"
$BLUE  = "$esc[34m"
$CYAN  = "$esc[36m"
$BOLD  = "$esc[1m"
$DIM   = "$esc[2m"
$NC    = "$esc[0m"

# ============================================================================
# Cleanup Handler
# ============================================================================
$TEMP_DIR = $null
Register-EngineEvent PowerShell.Exiting -Action {
    if ($TEMP_DIR -and (Test-Path $TEMP_DIR)) {
        Remove-Item -Recurse -Force $TEMP_DIR
    }
}

# ============================================================================
# Output Helpers (Direct symbols break tokenization)
# ============================================================================
function print_success { Write-Ansi "${GREEN}$([char]0x2713)${NC} $args" }  # ✓
function print_error   { Write-Ansi "${RED}$([char]0x2717)${NC} $args" }    # ✗
function print_warning { Write-Ansi "${YELLOW}$([char]0x0021)${NC} $args" }# !
function print_info    { Write-Ansi "${BLUE}$([char]0x2192)${NC} $args" }  # →


# Spinner
function Spin($job, $msg) {
    $spin = "|/-\"
    $i = 0
    while ($job.State -eq "Running") {
        $c = $spin[$i % $spin.Length]
        Write-Host "`r      $DIM$c$NC $msg" -NoNewline
        Start-Sleep -Milliseconds 100
        $i++
    }
    Write-Host "`r" -NoNewline
}

function print_banner {
    Write-Host ""
    Write-Ansi "  ${BOLD}Tiny${NC}${YELLOW}Torch${NC} ${DIM}v$TINYTORCH_VERSION${NC}"
    Write-Ansi "  ${DIM}Don't import it. Build it.${NC}"
    Write-Host ""
}


# ============================================================================
# Utility Functions
# ============================================================================
function Write-Ansi {
    param (
        [Parameter(Mandatory=$true)]
        [string] $Text
    )

    $esc = [char]27
    $ansiRegex = "$esc\[(\d+)m"

    # ansi to consolecolor map
    $ansiMap = @{
        31="Red"; 32="Green"; 33="Yellow"; 34="Blue"; 36="Cyan"
        0=$null    # reset
        1=$null    # bold (ignored)
        2=$null    # dim (ignored)
    }

    $lastIndex = 0
    $matches = [regex]::Matches($Text, $ansiRegex)

    foreach ($m in $matches) {
        $start = $m.Index
        $length = $m.Length
        $code = [int]$m.Groups[1].Value

        if ($start - $lastIndex -gt 0) {
            $chunk = $Text.Substring($lastIndex, $start - $lastIndex)
            Write-Host -NoNewline $chunk
        }

        # apply ansi
        if ($ansiMap.ContainsKey($code)) {
            if ($ansiMap[$code]) {
                [console]::ForegroundColor = $ansiMap[$code]
            } else {
                [console]::ResetColor()
            }
        }

        $lastIndex = $start + $length
    }

    # remaining
    if ($lastIndex -lt $Text.Length) {
        $chunk = $Text.Substring($lastIndex)
        Write-Host -NoNewline $chunk
    }

    [console]::ResetColor()
    Write-Host ""  # newline
}

function command_exists($cmd) {
    Get-Command $cmd -ErrorAction SilentlyContinue | Out-Null
    return $? -eq "True"
}

function get_python_cmd {
    # prefer python3, then python
    if (command_exists python3 -eq "True") { return "python3" }
    if (command_exists python -eq "True") { return "python" }
    return $null
}

function check_python_version($py) {
    $v = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $parts = $v.Split(".")
    if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 8) { return $v }
    throw "Python $v found, but 3.8+ required"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================
function check_write_permission {
    try {
        New-Item ".tinytorch_write_test" -ItemType File -Force | Out-Null
        Remove-Item ".tinytorch_write_test"
    } catch {
        print_error "Cannot write to this directory"
        exit 1
    }
}

function check_not_in_venv {
    if ($env:VIRTUAL_ENV) {
        print_warning "You're inside a virtual environment: $env:VIRTUAL_ENV"
    }
}

function check_internet {
    git ls-remote $REPO_URL | Out-Null
    print_success "GitHub reachable"
}

function check_prerequisites {
    if (command_exists git) {
        print_success "Git $(git --version)"
    } else {
        print_error "Git not found"
        exit 1
    }

    $PYTHON_CMD = get_python_cmd
    if (-not $PYTHON_CMD) {
        print_error "Python 3 not found"
        exit 1
    }

    $ver = check_python_version $PYTHON_CMD
    print_success "Python $ver"

    & $PYTHON_CMD -c "import venv" | Out-Null
    print_success "Python venv module"
}

function check_existing_directory {
    if (Test-Path $INSTALL_DIR) {
        print_error "Directory '$INSTALL_DIR' already exists"
        exit 1
    }
}

# ============================================================================
# Installation Steps
# ============================================================================
function prompt_install_directory {
    Write-Host ""
    Write-Host "Where would you like to install TinyTorch?"
    Write-Host "Press Enter for default: $PWD\tinytorch"
    $input = Read-Host "Install directory [tinytorch]"
    if ($input) { $script:INSTALL_DIR = $input }
}

function show_plan_and_confirm {
    Write-Host ""
    Write-Host "This will create: $PWD\$INSTALL_DIR"
    Write-Host "Source: $REPO_SHORT ($BRANCH)"
    Write-Host ""
}

function do_install {
    Write-Host "[1/4] Downloading from GitHub..."

    $TEMP_DIR = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath()) -Name ("tinytorch_" + [guid]::NewGuid())
    $repoPath = Join-Path $TEMP_DIR "repo"

    $job = Start-Job {
        git clone --depth 1 --filter=blob:none --sparse --branch $using:BRANCH $using:REPO_URL $using:repoPath 2>$null
    }


    Spin $job "Cloning repository..."
    Receive-Job $job | Out-Null

    Push-Location $repoPath
    git sparse-checkout set $SPARSE_PATH
    $COMMIT_HASH = git rev-parse --short HEAD
    Pop-Location

    Move-Item "$repoPath\$SPARSE_PATH" $INSTALL_DIR
    Remove-Item -Recurse -Force $TEMP_DIR
    $TEMP_DIR = $null

    print_success "Downloaded TinyTorch ($COMMIT_HASH)"

    Write-Host "[2/4] Creating Python environment..."
    Push-Location $INSTALL_DIR
    $PYTHON_CMD = get_python_cmd
    Write-Host "Using Python command: $PYTHON_CMD"
    & $PYTHON_CMD -m venv .venv
    .\.venv\Scripts\Activate.ps1
    print_success "Created virtual environment"

    Write-Host "[3/4] Installing dependencies..."

    # for some reason, pip doesnt like being run as a command here
    & $PYTHON_CMD -m pip install --upgrade pip | Out-Null
    & $PYTHON_CMD -m pip install -r requirements.txt | Out-Null
    & $PYTHON_CMD -m pip install -e . | Out-Null

    print_success "Installed dependencies"

    Write-Host "[4/4] Verifying installation..."
    if (Get-Command tito -ErrorAction SilentlyContinue) {
        print_success "Verified tito CLI"
    } else {
        print_warning "Activate venv before using tito"
    }

    Pop-Location
}

function print_success_message {
    Write-Host ""
    Write-Host "$([char]0x2713) TinyTorch installed successfully!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  cd $INSTALL_DIR"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  tito setup"
}

# ============================================================================
# Main
# ============================================================================
print_banner
check_write_permission
check_not_in_venv
check_prerequisites
check_internet
prompt_install_directory
check_existing_directory
show_plan_and_confirm

$reply = Read-Host "Continue? [Y/n]"
if ($reply -match "^[Nn]") { exit }

do_install
print_success_message
