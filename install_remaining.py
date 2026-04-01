# install_remaining.py
import subprocess
import sys

packages = [
    "python-dateutil==2.8.2",
    "boruta==0.3"
]

print("📦 Installing remaining packages...")
print("=" * 50)

for package in packages:
    print(f"\nInstalling {package}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ Successfully installed {package}")
    else:
        print(f"❌ Error: {result.stderr}")

print("\n" + "=" * 50)
print("✅ Installation complete!")