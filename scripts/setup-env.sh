#!/bin/bash

# Setup script for BioReasoning environment variables
# This script helps users configure API keys for podcast generation

echo "🔧 BioReasoning Environment Setup"
echo "=================================="
echo ""

# Check if .env file exists
if [[ -f ".env" ]]; then
    echo "📄 Found existing .env file"
    source .env
else
    echo "📄 Creating new .env file..."
    touch .env
fi

echo ""
echo "🔑 API Key Configuration"
echo "------------------------"

# Check OpenAI API Key
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "   Please get your API key from: https://platform.openai.com/api-keys"
    read -p "   Enter your OpenAI API key: " openai_key
    if [[ -n "$openai_key" ]]; then
        echo "OPENAI_API_KEY=$openai_key" >> .env
        export OPENAI_API_KEY="$openai_key"
        echo "   ✅ OpenAI API key saved"
    else
        echo "   ⚠️  Skipped OpenAI API key setup"
    fi
else
    echo "✅ OPENAI_API_KEY is set"
fi

# Check ElevenLabs API Key
if [[ -z "$ELEVENLABS_API_KEY" ]]; then
    echo "❌ ELEVENLABS_API_KEY not set"
    echo "   Please get your API key from: https://elevenlabs.io/speech-synthesis"
    read -p "   Enter your ElevenLabs API key: " elevenlabs_key
    if [[ -n "$elevenlabs_key" ]]; then
        echo "ELEVENLABS_API_KEY=$elevenlabs_key" >> .env
        export ELEVENLABS_API_KEY="$elevenlabs_key"
        echo "   ✅ ElevenLabs API key saved"
    else
        echo "   ⚠️  Skipped ElevenLabs API key setup"
    fi
else
    echo "✅ ELEVENLABS_API_KEY is set"
fi

echo ""
echo "🎙️  Podcast Generation Status"
echo "-----------------------------"

if [[ -n "$OPENAI_API_KEY" && -n "$ELEVENLABS_API_KEY" ]]; then
    echo "✅ Podcast generation is ready!"
    echo "   You can now use the 'Generate In-Depth Conversation' feature."
else
    echo "❌ Podcast generation is not available"
    echo "   Missing required API keys. Please run this script again to configure them."
fi

echo ""
echo "📝 Next Steps:"
echo "1. Run the application: ./scripts/run-knowledge-client.sh"
echo "2. Upload a document in the Documents page"
echo "3. Use the 'Generate In-Depth Conversation' button"
echo ""
echo "🔗 For help, visit: https://github.com/your-repo/bioreasoning" 