#!/usr/bin/env python3
"""
Fix for pydub regex warning in virtual environment.
This script patches the pydub utils.py file to fix the invalid escape sequence warning.
"""

import os
import re
import shutil
from pathlib import Path

def fix_pydub_regex():
    """Fix the invalid escape sequence in pydub utils.py"""
    
    # Find the pydub utils.py file in the virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found at .venv")
        return False
    
    pydub_utils_path = venv_path / "Lib" / "site-packages" / "pydub" / "utils.py"
    
    if not pydub_utils_path.exists():
        print(f"❌ pydub utils.py not found at {pydub_utils_path}")
        return False
    
    print(f"🔧 Fixing pydub regex in {pydub_utils_path}")
    
    # Read the file
    with open(pydub_utils_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the regex patterns that have invalid escape sequences
    # Pattern 1: '(dbl)p?( \(default\))?$' -> r'(dbl)p?( \(default\))?$'
    # Pattern 2: '(flt)p?( \(default\))?$' -> r'(flt)p?( \(default\))?$'
    
    original_patterns = [
        "'(dbl)p?( \\(default\\))?$'",
        "'(flt)p?( \\(default\\))?$'"
    ]
    
    fixed_patterns = [
        "r'(dbl)p?( \\(default\\))?$'",
        "r'(flt)p?( \\(default\\))?$'"
    ]
    
    modified = False
    for original, fixed in zip(original_patterns, fixed_patterns):
        if original in content:
            content = content.replace(original, fixed)
            modified = True
            print(f"✅ Fixed pattern: {original} -> {fixed}")
    
    if modified:
        # Create backup
        backup_path = pydub_utils_path.with_suffix('.py.backup')
        shutil.copy2(pydub_utils_path, backup_path)
        print(f"📦 Created backup at {backup_path}")
        
        # Write the fixed content
        with open(pydub_utils_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Successfully fixed pydub regex patterns")
        return True
    else:
        print("ℹ️  No regex patterns found to fix")
        return True

def install_ffmpeg_instructions():
    """Provide instructions for installing ffmpeg"""
    print("\n🔧 FFmpeg Installation Instructions:")
    print("=" * 50)
    print("To fix the 'Couldn't find ffmpeg or avconv' warning:")
    print()
    print("Windows:")
    print("1. Download ffmpeg from: https://ffmpeg.org/download.html")
    print("2. Extract to a folder (e.g., C:\\ffmpeg)")
    print("3. Add C:\\ffmpeg\\bin to your PATH environment variable")
    print("4. Restart your terminal/IDE")
    print()
    print("Alternative (using Chocolatey):")
    print("  choco install ffmpeg")
    print()
    print("Alternative (using winget):")
    print("  winget install ffmpeg")
    print()
    print("macOS:")
    print("  brew install ffmpeg")
    print()
    print("Linux (Ubuntu/Debian):")
    print("  sudo apt update && sudo apt install ffmpeg")
    print()
    print("After installation, restart your Python environment.")

if __name__ == "__main__":
    print("🔧 BioReasoning Fix Script")
    print("=" * 30)
    
    # Fix pydub regex
    success = fix_pydub_regex()
    
    # Provide ffmpeg instructions
    install_ffmpeg_instructions()
    
    if success:
        print("\n✅ Fix script completed successfully!")
        print("Please restart your Python environment for changes to take effect.")
    else:
        print("\n❌ Some fixes failed. Please check the output above.") 