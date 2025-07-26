#!/bin/bash

# Environment file checker utility
# This script helps diagnose issues with .env file configuration

echo "🔍 BioReasoning Environment Checker"
echo "==================================="
echo ""

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found!"
    echo "   Please create a .env file or run ./scripts/setup-env.sh"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Use Python to parse and validate .env file
python3 -c "
import os
import re
from pathlib import Path

def load_env_file(file_path):
    env_vars = {}
    errors = []
    warnings = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip()
            
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
                
            # Parse variable assignment
            if '=' in line:
                # Split on first = only
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                var_value = parts[1].strip() if len(parts) > 1 else ''
                
                # Validate variable name
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    # Remove quotes
                    if var_value.startswith('\"') and var_value.endswith('\"'):
                        var_value = var_value[1:-1]
                    elif var_value.startswith(\"'\") and var_value.endswith(\"'\"):
                        var_value = var_value[1:-1]
                    
                    env_vars[var_name] = var_value
                else:
                    errors.append(f'Line {line_num}: Invalid variable name \"{var_name}\"')
    
    return env_vars, errors, warnings

# Load and validate variables
env_vars, errors, warnings = load_env_file('.env')

print(f'📊 Environment Summary:')
print(f'   Total variables: {len(env_vars)}')
print(f'   Errors: {len(errors)}')
print(f'   Warnings: {len(warnings)}')
print('')

# Check for required variables
required_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
optional_vars = ['PHOENIX_API_KEY', 'PHOENIX_ENDPOINT', 'OTLP_ENDPOINT']

print('🔑 Required Variables:')
for var_name in required_vars:
    if var_name in env_vars:
        value = env_vars[var_name]
        if value:
            print(f'   ✅ {var_name}: {value[:20]}...')
        else:
            print(f'   ⚠️  {var_name}: Empty value')
    else:
        print(f'   ❌ {var_name}: Not found')

print('')
print('🔧 Optional Variables:')
for var_name in optional_vars:
    if var_name in env_vars:
        value = env_vars[var_name]
        print(f'   ✅ {var_name}: {value[:30]}...')
    else:
        print(f'   ⚪ {var_name}: Not set')

print('')

# Show errors if any
if errors:
    print('❌ Errors found:')
    for error in errors:
        print(f'   {error}')
    print('')

# Test podcast generation availability
openai_ok = 'OPENAI_API_KEY' in env_vars and env_vars['OPENAI_API_KEY']
elevenlabs_ok = 'ELEVENLABS_API_KEY' in env_vars and env_vars['ELEVENLABS_API_KEY']

print('🎙️  Podcast Generation Status:')
if openai_ok and elevenlabs_ok:
    print('   ✅ Ready - Both API keys are configured')
elif openai_ok:
    print('   ⚠️  Partially ready - Missing ELEVENLABS_API_KEY')
elif elevenlabs_ok:
    print('   ⚠️  Partially ready - Missing OPENAI_API_KEY')
else:
    print('   ❌ Not ready - Missing both API keys')

print('')
print('💡 Next Steps:')
if not openai_ok or not elevenlabs_ok:
    print('   1. Run ./scripts/setup-env.sh to configure missing API keys')
    print('   2. Get your API keys from:')
    print('      - OpenAI: https://platform.openai.com/api-keys')
    print('      - ElevenLabs: https://elevenlabs.io/speech-synthesis')
else:
    print('   1. Run ./scripts/run-knowledge-client.sh to start the application')
    print('   2. Upload a document and try podcast generation!')
" 2>/dev/null

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Environment check completed successfully!"
else
    echo ""
    echo "❌ Environment check failed. Please check your .env file format."
fi 