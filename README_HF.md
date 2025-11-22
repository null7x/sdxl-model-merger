---
title: SDXL Model Merger
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0
app_file: app.py
pinned: false
license: mit
tags:
  - stable-diffusion
  - sdxl
  - model-merging
  - image-generation
---

# SDXL Model Merger

Инструмент для объединения нескольких моделей Stable Diffusion XL с настраиваемыми весами.

## Особенности

- 🔀 Объединение нескольких SDXL моделей
- ⚙️ Настраиваемые веса для каждой модели
- 🛡️ Безопасная обработка несоответствий форм
- 📊 Подробный прогресс и отчёты
- 🚀 Поддержка GPU и CPU

## Использование

```bash
git clone https://huggingface.co/spaces/null7x/sdxl-model-merger
cd sdxl-model-merger

pip install -r requirements.txt

python merge.py
```

## Веса по умолчанию

- Jake Subway Surfer: 60%
- RealVisXL V50: 40%

Отредактируйте `merge.py` для изменения весов.

## Лицензия

MIT
