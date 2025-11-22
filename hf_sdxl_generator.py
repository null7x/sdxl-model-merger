"""
Gradio интерфейс для Stable Diffusion XL модели с Hugging Face
Оптимизирован для работы на CPU (Hugging Face Spaces free tier)

Автор: SDXL Model Merger
Лицензия: MIT
"""

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
from pathlib import Path


class SDXLGenerator:
    """Класс для управления генерацией изображений с SDXL моделью"""
    
    def __init__(self, model_id: str = "username/my-custom-model"):
        """
        Инициализирует генератор с моделью с Hugging Face
        
        Args:
            model_id: ID модели в формате "username/model-name"
        """
        self.model_id = model_id
        self.device = "cpu"  # Используем CPU для free tier HF Spaces
        self.pipe = None
        self.is_loaded = False
        
        # Пытаемся загрузить модель при инициализации
        self._load_model()
    
    def _load_model(self):
        """Загружает модель с Hugging Face с оптимизациями для CPU"""
        try:
            print(f"📥 Загрузка модели: {self.model_id}...")
            print(f"   Устройство: {self.device}")
            
            # Загружаем pipeline с оптимизациями для CPU
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # float32 для стабильности на CPU
                use_safetensors=True,
                safety_checker=None,  # Отключаем для ускорения
                variant="fp32"
            )
            
            # Перемещаем модель на CPU
            self.pipe = self.pipe.to(self.device)
            
            # Оптимизация памяти для CPU
            self.pipe.enable_attention_slicing()  # Уменьшает использование памяти
            
            self.is_loaded = True
            print("✓ Модель успешно загружена!")
            
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            self.is_loaded = False
    
    def generate(self, prompt: str, negative_prompt: str = "", num_steps: int = 20) -> Image.Image:
        """
        Генерирует изображение по текстовому описанию
        
        Args:
            prompt: Текстовое описание изображения
            negative_prompt: То, что НЕ должно быть на изображении
            num_steps: Количество шагов денойзации (20-50, больше = качественнее)
            
        Returns:
            PIL Image объект или None если ошибка
        """
        
        if not self.is_loaded:
            return None, "❌ Модель не загружена. Проверьте ID модели и интернет-соединение."
        
        if not prompt or len(prompt.strip()) == 0:
            return None, "⚠️ Введите описание для изображения (prompt)"
        
        try:
            print(f"\n🎨 Генерирую изображение...")
            print(f"   Prompt: {prompt}")
            print(f"   Шагов: {num_steps}")
            
            # Генерируем изображение
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            print("✓ Изображение успешно сгенерировано!")
            return image, "✓ Готово!"
            
        except Exception as e:
            print(f"✗ Ошибка генерации: {e}")
            return None, f"❌ Ошибка: {str(e)}"


def create_gradio_interface():
    """
    Создаёт Gradio интерфейс для генератора
    
    Returns:
        gr.Blocks: Gradio интерфейс
    """
    
    # Инициализируем генератор
    # ИЗМЕНИТЕ ЭТО НА ВАШУ МОДЕЛЬ: "null7x/your-model"
    generator = SDXLGenerator(model_id="username/my-custom-model")
    
    # Создаём интерфейс
    with gr.Blocks(
        title="SDXL Генератор",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        css="""
        .title-text {
            text-align: center;
            color: #1f77b4;
        }
        """
    ) as demo:
        
        # Заголовок и описание
        gr.Markdown("# 🎨 Моя SDXL модель")
        gr.Markdown("## Генератор изображений 24/7 с моей кастомной моделью")
        
        # Информация о статусе
        if generator.is_loaded:
            gr.Markdown("✅ **Модель загружена и готова к использованию!**")
        else:
            gr.Markdown("❌ **Ошибка загрузки модели. Проверьте ID и интернет.**")
        
        with gr.Row():
            with gr.Column(scale=1):
                # ============ ВВОД ДАННЫХ ============
                gr.Markdown("### 📝 Параметры генерации")
                
                # Основной промт (текстовое описание)
                prompt = gr.Textbox(
                    label="📋 Описание изображения (Prompt)",
                    placeholder="Напишите, что вы хотите увидеть на картинке...",
                    lines=4,
                    interactive=True
                )
                
                # Отрицательный промт (что исключить)
                negative_prompt = gr.Textbox(
                    label="🚫 Что исключить (Negative Prompt)",
                    placeholder="низкое качество, размытое, искажённое...",
                    lines=2,
                    interactive=True,
                    value="low quality, blurry, distorted"
                )
                
                # Количество шагов
                num_steps = gr.Slider(
                    label="⚙️ Количество шагов (качество/скорость)",
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=1,
                    interactive=True
                )
                
                gr.Markdown("💡 **Совет:** Больше шагов = лучше качество, но медленнее")
                
                # Кнопка генерации
                generate_btn = gr.Button(
                    "🚀 Генерировать изображение",
                    variant="primary",
                    size="lg",
                    interactive=generator.is_loaded
                )
            
            with gr.Column(scale=1):
                # ============ ВЫВОД ДАННЫХ ============
                gr.Markdown("### 🖼️ Результат")
                
                # Генерируемое изображение
                output_image = gr.Image(
                    label="Сгенерированное изображение",
                    type="pil",
                    interactive=False
                )
                
                # Статус сообщение
                status_text = gr.Textbox(
                    label="Статус",
                    interactive=False,
                    lines=1
                )
                
                # Кнопка скачивания (автоматически появляется при генерации)
                download_btn = gr.DownloadButton(
                    label="⬇️ Скачать изображение",
                    interactive=False,
                    visible=False
                )
        
        # ============ ЛОГИКА СОБЫТИЙ ============
        
        def on_generate(prompt_text, neg_prompt, steps):
            """Обработчик клика на кнопку генерации"""
            
            # Генерируем изображение
            image, status = generator.generate(
                prompt=prompt_text,
                negative_prompt=neg_prompt,
                num_steps=int(steps)
            )
            
            # Обновляем интерфейс
            outputs = {
                output_image: image,
                status_text: status,
            }
            
            # Если генерация успешна, активируем кнопку скачивания
            if image is not None:
                # Сохраняем временный файл для скачивания
                temp_path = "/tmp/generated_image.png"
                image.save(temp_path)
                outputs[download_btn] = temp_path
                outputs[download_btn.visible] = True
            
            return image, status
        
        # Подключаем событие клика кнопки
        generate_btn.click(
            fn=on_generate,
            inputs=[prompt, negative_prompt, num_steps],
            outputs=[output_image, status_text, download_btn]
        )
        
        # ============ ПРИМЕРЫ ============
        gr.Examples(
            examples=[
                [
                    "красивый пейзаж с горами и закатом, фотореалистичный, 4k",
                    "низкое качество, размытое",
                    20
                ],
                [
                    "научно-фантастический город ночью с неоновыми огнями, киберпанк стиль",
                    "размытое, дневное время",
                    25
                ],
                [
                    "портрет красивой девушки, детальный, высокое качество",
                    "уродливое, искажённое лицо",
                    20
                ],
            ],
            inputs=[prompt, negative_prompt, num_steps],
            outputs=[output_image, status_text],
            fn=on_generate,
            cache_examples=False
        )
        
        # ============ ИНФОРМАЦИЯ ============
        gr.Markdown("---")
        gr.Markdown("""
        ### ℹ️ Информация
        - **Модель:** SDXL (Stable Diffusion XL)
        - **Устройство:** CPU (работает на free tier HF Spaces)
        - **Время генерации:** 5-15 минут (зависит от количества шагов)
        - **Разрешение:** 512x512 пикселей
        
        ### 🔧 Как использовать
        1. Напишите описание в поле "Описание изображения"
        2. Опционально, укажите что исключить в "Отрицательный промт"
        3. Установите количество шагов (20-50 рекомендуется)
        4. Нажмите "Генерировать изображение"
        5. Скачайте результат если доволены
        
        ### 📚 Ссылки
        - [GitHub](https://github.com/null7x/sdxl-model-merger)
        - [Hugging Face](https://huggingface.co/spaces/Aminjon2005/sdxl-model-merger)
        """)
    
    return demo


if __name__ == "__main__":
    # Создаём интерфейс
    interface = create_gradio_interface()
    
    # Запускаем на адресе 0.0.0.0 для доступа извне (для HF Spaces)
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Для local testing
        show_error=True,
        debug=False
    )
