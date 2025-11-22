"""
Генератор изображений с использованием объединённой SDXL модели
Требует: diffusers, transformers, safetensors
"""

import torch
from pathlib import Path


def test_generation():
    """Тест генерации с объединённой моделью"""
    
    print("=" * 60)
    print("Генератор изображений SDXL")
    print("=" * 60)
    
    # Проверяем модель
    model_path = Path("models/merged_sdxl.safetensors")
    if not model_path.exists():
        print(f"\n✗ Ошибка: Модель не найдена в {model_path}")
        return False
    
    print(f"\n✓ Модель найдена: {model_path}")
    print(f"  Размер: {model_path.stat().st_size / (1024**3):.2f} ГБ")
    
    # Проверяем требуемые пакеты
    print("\n[1/3] Проверка зависимостей...")
    
    required_packages = {
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'safetensors': 'safetensors',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name} установлен")
        except ImportError:
            print(f"  ✗ {pip_name} НЕ установлен")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠ Требуется установить пакеты:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        return False
    
    print("\n[2/3] Загрузка модели...")
    
    try:
        from diffusers import StableDiffusionXLPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Используется устройство: {device}")
        
        # Загружаем pipeline с нашей объединённой моделью
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        print("  ✓ Модель успешно загружена в pipeline")
        
        print("\n[3/3] Генерация тестового изображения...")
        print("  (Это может занять несколько минут на CPU)")
        
        prompt = "a beautiful landscape with mountains and sunset, photorealistic, 4k"
        negative_prompt = "blurry, low quality, distorted"
        
        print(f"\n  Prompt: {prompt}")
        print(f"  Negative: {negative_prompt}")
        
        # Генерируем изображение
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        # Сохраняем результат
        output_path = Path("outputs/test_generation.png")
        output_path.parent.mkdir(exist_ok=True)
        image.save(output_path)
        
        print(f"\n  ✓ Изображение сохранено: {output_path}")
        
        print("\n" + "=" * 60)
        print("✓ ГЕНЕРАЦИЯ УСПЕШНА!")
        print("=" * 60)
        print("\nМодель работает корректно и готова к использованию!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Ошибка при генерации:")
        print(f"  {e}")
        return False


if __name__ == "__main__":
    success = test_generation()
    exit(0 if success else 1)
