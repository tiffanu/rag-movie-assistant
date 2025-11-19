# RAG Movie Assistant
**Анализатор кинопредпочтений с глубоким поиском**

Ассистент для подбора кино по описанию, работающий на RAG-архитектуре с локальными моделями.

### Архитектура
- **Эмбеддинги**: `mistral-embed`
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2`
- **UI**: Streamlit

### Тестирование

- Датасет искуственно сгенерированный с помощью модели `gpt-4o-mini`
- Оценка Answer_v_reference_score - сравнение with/without RAG

### Быстрый старт

1. Клонируем репозиторий
   ```bash
   git clone <repo>
   cd rag-movie-assistant
   
2. Устанавливаем зависимости
   ```bash
   pip install -r requirements.txt
   
3. Вставляем Mistral API Key в .env.example
   ```
   MISTRAL_API_KEY=your_mistral_api_key

4. Добавляем API ключ в окружение
   ```bash
   cp .env.example .env

5. Запуск
   ```bash
   streamlit run app_streamlit.py

Первый запуск скачает модели и загрузит базу (~1–2 минуты)
