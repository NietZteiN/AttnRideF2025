"""
Interactive HuggingFace Login Script
Helps you authenticate with HuggingFace to access Llama 3 models.
"""

from huggingface_hub import login
import sys

print("=" * 60)
print("HuggingFace Login")
print("=" * 60)
print()
print("To get your access token:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Click 'New token' or copy an existing token")
print("3. Make sure it has 'read' permissions")
print("4. Paste it below when prompted")
print()
print("IMPORTANT: Also request access to Llama 3:")
print("   https://huggingface.co/meta-llama/Meta-Llama-3-8B")
print()
print("=" * 60)
print()

try:
    login()
    print()
    print("✓ Successfully logged in to HuggingFace!")
    print()
    print("Next steps:")
    print("1. Make sure you have access to Llama 3:")
    print("   https://huggingface.co/meta-llama/Meta-Llama-3-8B")
    print()
    print("2. Run the extraction script:")
    print("   python extract_llama3_data.py")
    print()
    
except Exception as e:
    print()
    print("✗ Login failed!")
    print(f"Error: {e}")
    print()
    print("Please make sure:")
    print("- Your token is valid (check https://huggingface.co/settings/tokens)")
    print("- You copied the entire token (no spaces)")
    print("- You have internet connection")
    sys.exit(1)
