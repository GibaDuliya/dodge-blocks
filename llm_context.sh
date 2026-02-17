#!/bin/bash

# Имя выходного файла
OUTPUT="full_project_context.txt"

# Очищаем файл
echo "PROJECT CONTEXT" > "$OUTPUT"
echo "===============" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# 1. СТРУКТУРА ПРОЕКТА
# Показываем дерево, игнорируя venv, git, cache, artifacts
echo "--- DIRECTORY TREE ---" >> "$OUTPUT"
if command -v tree &> /dev/null; then
    # -I игнорирует папки по паттерну
    tree -I "venv|__pycache__|.git|artifacts|*.egg-info|.pytest_cache" >> "$OUTPUT"
else
    # Fallback если нет tree
    find . -maxdepth 3 -not -path '*/.*' -not -path './venv*' -not -path './artifacts*' >> "$OUTPUT"
fi
echo "" >> "$OUTPUT"

# 2. СОДЕРЖИМОЕ ФАЙЛОВ
echo "--- FILE CONTENTS ---" >> "$OUTPUT"

# Ищем файлы:
# - Расширения: .py, .sh, .md, requirements.txt
# - Исключаем: venv, .git, artifacts, __pycache__
# - Исключаем конкретно: __init__.py, сам выходной файл
find . -type f \
    \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "requirements.txt" \) \
    -not -path "*/venv/*" \
    -not -path "*/.git/*" \
    -not -path "*/artifacts/*" \
    -not -path "*/__pycache__/*" \
    -not -name "__init__.py" \
    -not -name "$OUTPUT" \
    | sort | while read -r file; do

    echo "================================================" >> "$OUTPUT"
    echo "FILE: $file" >> "$OUTPUT"
    echo "================================================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n\n" >> "$OUTPUT"
    
    echo "Added: $file"
done

echo "------------------------------------------------"
echo "Context generated in: $OUTPUT"