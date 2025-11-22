#!/usr/bin/env python
"""
Загрузить проект на Hugging Face Spaces
"""

from huggingface_hub import upload_folder
from pathlib import Path
import os

# ИСПОЛЬЗУЙТЕ ПЕРЕМЕННУЮ ОКРУЖЕНИЯ ДЛЯ ТОКЕНА!
# Установите токен через: huggingface-cli login
# Или через переменную: export HF_TOKEN="your-token"

token = os.getenv("HF_TOKEN")  # Получаем из переменной окружения
repo_id = "username/your-space-name"  # ИЗМЕНИТЕ НА ВАШУ!

# Файлы для загрузки
files_to_upload = [
    "app.py",
    "merge.py", 
    "test_model.py",
    "generate_image.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

project_dir = Path(".")

try:
    print("Загрузка файлов на Hugging Face Spaces...")
    
    upload_folder(
        folder_path=str(project_dir),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        allow_patterns=[f for f in files_to_upload],
        commit_message="Initial commit: SDXL Model Merger"
    )
    
    print(f"✓ Успешно загружено на https://huggingface.co/spaces/{repo_id}")
    
except Exception as e:
    print(f"✗ Ошибка загрузки: {e}")
    print(f"\nПроверьте:")
    print(f"1. Токен валиден: {token[:10]}...")
    print(f"2. repo_id правильный: {repo_id}")
    print(f"3. Space создан вручную на https://huggingface.co/spaces/create")
