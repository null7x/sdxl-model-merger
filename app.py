"""
Gradio интерфейс для SDXL Model Merger
Предназначен для запуска на Hugging Face Spaces
"""

import gradio as gr
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import os


class SDXLMergerInterface:
    """Интерфейс для объединения моделей через Gradio"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def merge_models(self, model1_weight: float, model2_weight: float, progress=gr.Progress()):
        """
        Объединяет две модели с указанными весами
        
        Args:
            model1_weight: Вес первой модели (0-1)
            model2_weight: Вес второй модели (0-1)
            
        Returns:
            Статус и информация о объединении
        """
        
        try:
            # Нормализуем веса
            total = model1_weight + model2_weight
            if total == 0:
                return "❌ Ошибка: Оба веса не могут быть 0", None
            
            w1 = model1_weight / total
            w2 = model2_weight / total
            
            progress(0, "Проверка наличия моделей...")
            
            # Ищем доступные модели
            safetensors_files = list(self.models_dir.glob("*.safetensors"))
            
            if len(safetensors_files) < 2:
                return (
                    f"❌ Ошибка: Нужно минимум 2 модели, найдено {len(safetensors_files)}\n"
                    f"Загруженные модели: {[f.name for f in safetensors_files]}",
                    None
                )
            
            # Используем первые две модели
            model1_path = safetensors_files[0]
            model2_path = safetensors_files[1]
            
            progress(0.2, f"Загрузка {model1_path.name}...")
            model1 = load_file(str(model1_path))
            
            progress(0.4, f"Загрузка {model2_path.name}...")
            model2 = load_file(str(model2_path))
            
            progress(0.6, "Объединение моделей...")
            
            # Объединяем модели
            merged = {}
            all_keys = set(model1.keys()) | set(model2.keys())
            
            for i, key in enumerate(sorted(all_keys)):
                if key in model1 and key in model2:
                    # Проверяем совместимость форм
                    if model1[key].shape == model2[key].shape:
                        merged[key] = (
                            model1[key].float() * w1 + 
                            model2[key].float() * w2
                        )
                    else:
                        # Пропускаем несовместимые ключи
                        merged[key] = model1[key]
                elif key in model1:
                    merged[key] = model1[key]
                else:
                    merged[key] = model2[key]
                
                if i % 100 == 0:
                    progress(0.6 + (i / len(all_keys)) * 0.3, f"Обработано {i}/{len(all_keys)} ключей")
            
            # Сохраняем результат
            progress(0.9, "Сохранение объединённой модели...")
            output_path = self.models_dir / "merged_model.safetensors"
            save_file(merged, str(output_path))
            
            file_size_gb = output_path.stat().st_size / (1024**3)
            
            result_text = (
                f"✅ Объединение успешно завершено!\n\n"
                f"Модель 1: {model1_path.name} ({w1:.0%})\n"
                f"Модель 2: {model2_path.name} ({w2:.0%})\n\n"
                f"Выходной файл: {output_path.name}\n"
                f"Размер: {file_size_gb:.2f} ГБ\n"
                f"Объединено ключей: {len(merged)}"
            )
            
            progress(1.0, "Завершено!")
            
            return result_text, str(output_path)
            
        except Exception as e:
            return f"❌ Ошибка: {str(e)}", None


def create_interface():
    """Создаёт Gradio интерфейс"""
    
    merger = SDXLMergerInterface()
    
    with gr.Blocks(title="SDXL Model Merger") as demo:
        gr.Markdown("""
        # 🎨 SDXL Model Merger
        
        Объединяйте несколько моделей Stable Diffusion XL с настраиваемыми весами.
        
        ## Инструкция:
        1. Загрузите модели в файловую систему (форматом `.safetensors`)
        2. Установите веса для каждой модели
        3. Нажмите "Объединить модели"
        4. Загрузите результат
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Параметры объединения")
                
                weight1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.6,
                    step=0.05,
                    label="Вес модели 1"
                )
                
                weight2 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.4,
                    step=0.05,
                    label="Вес модели 2"
                )
                
                merge_btn = gr.Button("🔀 Объединить модели", variant="primary", size="lg")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Результат",
                    lines=10,
                    interactive=False
                )
                
                output_file = gr.File(
                    label="Скачать объединённую модель",
                    visible=False
                )
        
        def on_merge(w1, w2):
            text, file_path = merger.merge_models(w1, w2)
            # Возвращаем обновления для компонентов
            updates = {
                output_text: text,
            }
            if file_path:
                updates[output_file] = file_path
            return text, file_path if file_path else None
        
        merge_btn.click(
            fn=on_merge,
            inputs=[weight1, weight2],
            outputs=[output_text, output_file]
        )
        
        gr.Markdown("""
        ### Информация
        - Текущее устройство: {} 
        - Формат моделей: SafeTensors (.safetensors)
        - Поддерживаемые версии: SDXL 1.0+
        
        [GitHub](https://github.com/null7x/sdxl-model-merger) | 
        [Документация](https://github.com/null7x/sdxl-model-merger/blob/main/README.md)
        """.format("🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"))
    
    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
