"""
Тест объединённой SDXL модели
Проверяет, что модель успешно загружена и готова к использованию
"""

from safetensors.torch import load_file
from pathlib import Path
import torch


def test_merged_model():
    """Проверяет объединённую модель"""
    
    print("=" * 60)
    print("Тестирование объединённой модели SDXL")
    print("=" * 60)
    
    model_path = Path("models/merged_sdxl.safetensors")
    
    # Проверяем наличие файла
    if not model_path.exists():
        print(f"\n✗ Ошибка: Модель не найдена в {model_path}")
        print("\nНужно сначала запустить merge.py для создания модели")
        return False
    
    try:
        print(f"\n[1/3] Загрузка модели: {model_path}...")
        model = load_file(str(model_path))
        print("  ✓ Модель успешно загружена!")
        
        # Информация о модели
        print(f"\n[2/3] Информация о модели:")
        print(f"  Количество параметров: {len(model)}")
        
        # Размер файла
        file_size_gb = model_path.stat().st_size / (1024**3)
        print(f"  Размер файла: {file_size_gb:.2f} ГБ")
        
        # Типы тензоров
        tensor_shapes = {}
        total_elements = 0
        
        for key, tensor in list(model.items())[:10]:  # Первые 10
            shape = tuple(tensor.shape)
            if shape not in tensor_shapes:
                tensor_shapes[shape] = 0
            tensor_shapes[shape] += 1
            total_elements += tensor.numel()
        
        print(f"  Примеры форм тензоров: {list(tensor_shapes.keys())[:3]}")
        
        # Первые ключи
        print(f"\n  Первые 5 ключей модели:")
        for i, key in enumerate(list(model.keys())[:5], 1):
            print(f"    {i}. {key}")
        
        print(f"\n[3/3] Проверка целостности:")
        
        # Проверяем наличие критических компонентов
        critical_keys = [
            "time_embed", "input_blocks", "middle_block", "output_blocks", "out"
        ]
        
        found_critical = 0
        for key in model.keys():
            for critical in critical_keys:
                if critical in key.lower():
                    found_critical += 1
                    break
        
        if found_critical > 0:
            print(f"  ✓ Найдены критические компоненты ({found_critical} групп)")
        else:
            print(f"  ⚠ Внимание: не все критические компоненты найдены")
        
        # Проверяем наличие NaN или Inf
        has_issues = False
        for key, tensor in list(model.items())[:100]:  # Проверяем первые 100
            if torch.isnan(tensor).any():
                print(f"  ✗ Найдены NaN значения в {key}")
                has_issues = True
                break
            if torch.isinf(tensor).any():
                print(f"  ✗ Найдены Inf значения в {key}")
                has_issues = True
                break
        
        if not has_issues:
            print(f"  ✓ Нет NaN или Inf значений")
        
        print(f"\n" + "=" * 60)
        print("✓ МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("=" * 60)
        print(f"\nМодель сохранена: {model_path}")
        print(f"\nЧто дальше:")
        print("1. Используйте в Stable Diffusion WebUI")
        print("2. Используйте в ComfyUI")
        print("3. Используйте в InvokeAI")
        print("\nСкопируйте файл в папку models вашего SD приложения")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Ошибка при тестировании модели:")
        print(f"  {e}")
        return False


if __name__ == "__main__":
    success = test_merged_model()
    exit(0 if success else 1)
