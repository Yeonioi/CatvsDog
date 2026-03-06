"""
Download and organize the Microsoft Cats and Dogs dataset
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Dataset setup
DATASET_DIR = "data"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def create_directories():
    """Create necessary directories."""
    os.makedirs(os.path.join(TRAIN_DIR, "cats"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, "dogs"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "cats"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "dogs"), exist_ok=True)
    print("✓ Created directory structure")

def download_dataset():
    """Download the Microsoft Cats and Dogs dataset."""
    print("\n📥 Downloading Microsoft Cats and Dogs dataset...")
    print("This may take a few minutes (~700 MB)...")
    
    # Official Microsoft dataset URL
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_path = "cats_and_dogs.zip"
    
    try:
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 // total_size, 100)
            print(f"\rProgress: {percent}%", end="")
        
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\n✓ Download complete!")
        return zip_path
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nManual download option:")
        print("1. Visit: https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765")
        print("2. Download and extract to 'PetImages' folder")
        print("3. The script will organize the files")
        return None

def extract_dataset(zip_path):
    """Extract the dataset."""
    if not os.path.exists(zip_path):
        print(f"✗ {zip_path} not found")
        return False
    
    print("\n📦 Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✓ Extraction complete!")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def organize_dataset():
    """Organize images into train/test cat/dog folders."""
    print("\n📂 Organizing dataset...")
    
    pet_images_dir = "PetImages"
    
    if not os.path.exists(pet_images_dir):
        print(f"✗ {pet_images_dir} folder not found")
        return False
    
    # Process cats
    cats_dir = os.path.join(pet_images_dir, "Cat")
    if os.path.exists(cats_dir):
        cat_files = [f for f in os.listdir(cats_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        split = int(len(cat_files) * 0.8)
        
        for i, file in enumerate(cat_files):
            src = os.path.join(cats_dir, file)
            try:
                if i < split:
                    dst = os.path.join(TRAIN_DIR, "cats", file)
                else:
                    dst = os.path.join(TEST_DIR, "cats", file)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"  ⚠ Skipped {file}: {str(e)[:50]}")
        
        print(f"  ✓ Cats: {split} train, {len(cat_files) - split} test")
    
    # Process dogs
    dogs_dir = os.path.join(pet_images_dir, "Dog")
    if os.path.exists(dogs_dir):
        dog_files = [f for f in os.listdir(dogs_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        split = int(len(dog_files) * 0.8)
        
        for i, file in enumerate(dog_files):
            src = os.path.join(dogs_dir, file)
            try:
                if i < split:
                    dst = os.path.join(TRAIN_DIR, "dogs", file)
                else:
                    dst = os.path.join(TEST_DIR, "dogs", file)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"  ⚠ Skipped {file}: {str(e)[:50]}")
        
        print(f"  ✓ Dogs: {split} train, {len(dog_files) - split} test")
    
    return True

def count_images():
    """Display image counts."""
    print("\n📊 Dataset Summary:")
    
    train_cats = len(os.listdir(os.path.join(TRAIN_DIR, "cats")))
    train_dogs = len(os.listdir(os.path.join(TRAIN_DIR, "dogs")))
    test_cats = len(os.listdir(os.path.join(TEST_DIR, "cats")))
    test_dogs = len(os.listdir(os.path.join(TEST_DIR, "dogs")))
    
    print(f"  Training: {train_cats} cats, {train_dogs} dogs")
    print(f"  Testing:  {test_cats} cats, {test_dogs} dogs")
    print(f"  Total:    {train_cats + test_cats + train_dogs + test_dogs} images")

def cleanup(zip_path):
    """Clean up temporary files."""
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"\n✓ Cleaned up {zip_path}")

def main():
    print("=" * 60)
    print("Microsoft Cats and Dogs Dataset Downloader")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Download dataset
    zip_path = download_dataset()
    if not zip_path:
        print("\nAlternatively, manually download from:")
        print("https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765")
        return
    
    # Extract dataset
    if not extract_dataset(zip_path):
        return
    
    # Organize dataset
    if not organize_dataset():
        return
    
    # Show summary
    count_images()
    
    # Cleanup
    cleanup(zip_path)
    
    print("\n" + "=" * 60)
    print("✓ Dataset ready for training!")
    print("=" * 60)
    print("\nNext step: Update train.py to use the 'data' folder")
    print("Then run: .venv\\Scripts\\python.exe train.py")

if __name__ == "__main__":
    main()
