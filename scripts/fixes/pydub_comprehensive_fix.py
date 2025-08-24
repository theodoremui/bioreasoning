#!/usr/bin/env python3
"""
Comprehensive fix for pydub regex warnings in virtual environment.
This script patches the pydub utils.py file to fix all invalid escape sequence warnings.
"""

import os
import re
import shutil
from pathlib import Path


def fix_pydub_regex_comprehensive():
    """Fix all invalid escape sequences in pydub utils.py"""

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
    with open(pydub_utils_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define all the regex patterns that need fixing
    fixes = [
        # Line 300: '([su]([0-9]{1,2})p?) \(([0-9]{1,2}) bit\)$' -> r'([su]([0-9]{1,2})p?) \(([0-9]{1,2}) bit\)$'
        (
            "'([su]([0-9]{1,2})p?) \\(([0-9]{1,2}) bit\\)\\$'",
            "r'([su]([0-9]{1,2})p?) \\(([0-9]{1,2}) bit\\)\\$'",
        ),
        # Line 301: '([su]([0-9]{1,2})p?)( \(default\))?$' -> r'([su]([0-9]{1,2})p?)( \(default\))?$'
        (
            "'([su]([0-9]{1,2})p?)( \\(default\\))?\\$'",
            "r'([su]([0-9]{1,2})p?)( \\(default\\))?\\$'",
        ),
        # Line 310: '(flt)p?( \(default\))?$' -> r'(flt)p?( \(default\))?$'
        ("'(flt)p?( \\(default\\))?\\$'", "r'(flt)p?( \\(default\\))?\\$'"),
        # Line 314: '(dbl)p?( \(default\))?$' -> r'(dbl)p?( \(default\))?$'
        ("'(dbl)p?( \\(default\\))?\\$'", "r'(dbl)p?( \\(default\\))?\\$'"),
    ]

    # Also fix any malformed patterns that might have been created
    malformed_fixes = [
        ("rr'(flt)p?( \\(default\\))?\\$'", "r'(flt)p?( \\(default\\))?\\$'"),
        ("rr'(dbl)p?( \\(default\\))?\\$'", "r'(dbl)p?( \\(default\\))?\\$'"),
    ]

    modified = False

    # Apply the main fixes
    for original, fixed in fixes:
        if original in content:
            content = content.replace(original, fixed)
            modified = True
            print(f"✅ Fixed pattern: {original} -> {fixed}")

    # Apply malformed fixes
    for original, fixed in malformed_fixes:
        if original in content:
            content = content.replace(original, fixed)
            modified = True
            print(f"✅ Fixed malformed pattern: {original} -> {fixed}")

    if modified:
        # Create backup
        backup_path = pydub_utils_path.with_suffix(".py.backup")
        shutil.copy2(pydub_utils_path, backup_path)
        print(f"📦 Created backup at {backup_path}")

        # Write the fixed content
        with open(pydub_utils_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Successfully fixed all pydub regex patterns")
        return True
    else:
        print("ℹ️  No regex patterns found to fix")
        return True


def test_pydub_import():
    """Test if pydub can be imported without warnings"""
    try:
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import pydub

            print(f"✅ Pydub imported successfully (version: {pydub.__version__})")

            if w:
                print("⚠️  Warnings found:")
                for warning in w:
                    print(f"   {warning.message}")
                return False
            else:
                print("✅ No warnings during pydub import")
                return True
    except Exception as e:
        print(f"❌ Error importing pydub: {e}")
        return False


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
    print("🔧 BioReasoning Comprehensive Fix Script")
    print("=" * 40)

    # Fix pydub regex
    success = fix_pydub_regex_comprehensive()

    if success:
        print("\n🧪 Testing pydub import...")
        test_success = test_pydub_import()

        if test_success:
            print("✅ All fixes applied successfully!")
        else:
            print("⚠️  Some issues remain. Check the warnings above.")

    # Provide ffmpeg instructions
    install_ffmpeg_instructions()

    if success:
        print("\n✅ Fix script completed!")
        print("Please restart your Python environment for changes to take effect.")
    else:
        print("\n❌ Some fixes failed. Please check the output above.")
