"""
Объединитель моделей Stable Diffusion XL
Объединяет несколько моделей SDXL с использованием взвешенной суммы
"""

import os
import torch
import safetensors
from safetensors.torch import load_file, save_file
from pathlib import Path
from typing import Dict, Tuple
import sys


class SDXLModelMerger:
    """Объединяет несколько моделей SDXL с взвешенным усреднением"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Инициализация объединителя
        
        Args:
            models_dir: Директория с файлами моделей
        """
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
    
    def load_model(self, model_name: str) -> Dict:
        """
        Загружает модель из файла safetensors
        
        Args:
            model_name: Имя файла модели
            
        Returns:
            Словарь, содержащий состояние модели
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        print(f"Загрузка модели: {model_name}...")
        try:
            state_dict = load_file(str(model_path))
            print(f"  ✓ Загружено {len(state_dict)} ключей")
            return state_dict
        except Exception as e:
            print(f"  ✗ Ошибка загрузки модели: {e}")
            raise
    
    def merge_models(
        self,
        models: Dict[str, Tuple[str, float]],
        output_path: str = "merged_sdxl.safetensors"
    ) -> None:
        """
        Объединяет несколько моделей с указанными весами
        
        Args:
            models: Словарь с именами моделей и их (имя файла, вес) кортежами
            output_path: Путь для сохранения объединённой модели
            
        Пример:
            models = {
                "juggernaut": ("juggernautXL_v8.safetensors", 0.55),
                "realvis": ("realvisxlV40.safetensors", 0.25),
                "zavymix": ("zavy_mix_xl.safetensors", 0.20),
            }
        """
        
        # Проверяем, что веса в сумме дают примерно 1.0
        total_weight = sum(weight for _, weight in models.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠ Предупреждение: Веса в сумме дают {total_weight:.2f}, а не 1.0")
        
        # Загружаем все модели
        loaded_models = {}
        for model_key, (model_file, weight) in models.items():
            print(f"\n[1/3] Загрузка {model_key}...")
            loaded_models[model_key] = {
                "state_dict": self.load_model(model_file),
                "weight": weight
            }
        
        # Получаем все уникальные ключи из всех моделей
        all_keys = set()
        for model_data in loaded_models.values():
            all_keys.update(model_data["state_dict"].keys())
        
        print(f"\n[2/3] Объединение {len(all_keys)} ключей...")
        
        merged_state_dict = {}
        skipped_keys = []
        merged_count = 0
        
        for i, key in enumerate(sorted(all_keys), 1):
            # Индикатор прогресса
            if i % 100 == 0:
                print(f"  Обработано {i}/{len(all_keys)} ключей ({merged_count} объединено, {len(skipped_keys)} пропущено)")
            
            # Собираем тензоры из моделей, которые имеют этот ключ
            tensors_to_merge = []
            weights_to_use = []
            
            for model_key, model_data in loaded_models.items():
                if key in model_data["state_dict"]:
                    tensors_to_merge.append(model_data["state_dict"][key])
                    weights_to_use.append(model_data["weight"])
            
            # Пропускаем, если ключа нет во всех моделях (опционально)
            if len(tensors_to_merge) < len(loaded_models):
                # Объединяем, если ключ есть хотя бы в одной модели
                pass
            
            # Проверяем несоответствия форм
            try:
                first_shape = tensors_to_merge[0].shape
                shape_mismatch = False
                
                for tensor in tensors_to_merge[1:]:
                    if tensor.shape != first_shape:
                        shape_mismatch = True
                        break
                
                if shape_mismatch:
                    print(f"  ⚠ Пропуск {key}: несоответствие форм")
                    skipped_keys.append(key)
                    continue
                
                # Выполняем взвешенную сумму
                merged_tensor = None
                for tensor, weight in zip(tensors_to_merge, weights_to_use):
                    if merged_tensor is None:
                        merged_tensor = tensor.float() * weight
                    else:
                        merged_tensor += tensor.float() * weight
                
                # Нормализуем веса
                total_weight_used = sum(weights_to_use)
                if total_weight_used > 0:
                    merged_tensor = merged_tensor / total_weight_used
                
                merged_state_dict[key] = merged_tensor
                merged_count += 1
                
            except Exception as e:
                print(f"  ✗ Ошибка объединения {key}: {e}")
                skipped_keys.append(key)
        
        print(f"\n  ✓ Успешно объединено {merged_count} ключей")
        if skipped_keys:
            print(f"  ⚠ Пропущено {len(skipped_keys)} ключей из-за несоответствия форм или ошибок")
        
        # Сохраняем объединённую модель
        print(f"\n[3/3] Сохранение объединённой модели в {output_path}...")
        
        try:
            output_full_path = self.models_dir / output_path
            output_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_file(merged_state_dict, str(output_full_path))
            print(f"  ✓ Успешно сохранено в {output_full_path}")
            
            # Выводим размер файла
            file_size_gb = output_full_path.stat().st_size / (1024**3)
            print(f"  Размер файла: {file_size_gb:.2f} ГБ")
            
        except Exception as e:
            print(f"  ✗ Ошибка сохранения объединённой модели: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> None:
        """Выводит информацию о модели"""
        try:
            state_dict = self.load_model(model_name)
            print(f"\nМодель: {model_name}")
            print(f"Количество ключей: {len(state_dict)}")
            print(f"Примеры ключей: {list(state_dict.keys())[:5]}")
            
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"Приблизительное количество параметров: {total_params / 1e9:.2f}B")
        except Exception as e:
            print(f"Ошибка получения информации о модели: {e}")


def main():
    """Основная функция выполнения"""
    
    print("=" * 60)
    print("Объединитель моделей Stable Diffusion XL")
    print("=" * 60)
    
    # Конфигурация
    MODELS_DIR = "models"  # Директория, где хранятся модели
    
    MODELS_TO_MERGE = {
        "jake": ("JakeSubwaySurfer1-10 (4).safetensors", 0.60),
        "realvis": ("realvisxlV50_v50LightningBakedvae.safetensors", 0.40),
    }
    
    OUTPUT_FILE = "merged_sdxl.safetensors"
    
    # ============ НАСТРАИВАЕМЫЕ ВЕСА ============
    # Измените эти значения, чтобы изменить вклад моделей (должны суммироваться в ~1.0)
    # MODELS_TO_MERGE = {
    #     "jake": ("JakeSubwaySurfer1-10 (4).safetensors", 0.50),
    #     "realvis": ("realvisxlV50_v50LightningBakedvae.safetensors", 0.50),
    # }
    # =========================================
    
    try:
        # Создаём экземпляр объединителя
        merger = SDXLModelMerger(models_dir=MODELS_DIR)
        
        # Вывод информации о моделях (опционально - можно закомментировать)
        print("\nИнформация о моделях:")
        print("-" * 60)
        for model_key, (model_file, weight) in MODELS_TO_MERGE.items():
            print(f"{model_key}: {weight:.0%} веса")
        
        # Выполняем объединение
        print("\n" + "=" * 60)
        merger.merge_models(MODELS_TO_MERGE, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("✓ Объединение успешно завершено!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Объединение не удалось: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
