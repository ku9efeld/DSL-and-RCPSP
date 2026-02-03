
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional
import logging

# Отключаем логирование httpx и httpcore
logging.getLogger("httpx").setLevel(logging.WARNING)

class DeepSeekSession:
    def __init__(
        self,
        api_key: str,
        generate_prompt : str,
        system_prompt: str,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        temperature: float = 0.33,
        max_tokens: int = 4000,
        max_history: int = 20
    ):
        """
        Инициализация новой сессии с сохранением контекста.
        
        Args:
            api_key: API ключ DeepSeek
            base_url: URL API
            model_name: название модели
            system_prompt: системный промпт
            temperature: креативность ответов
            max_tokens: максимальное количество токенов в ответе
            max_history: максимальное количество сообщений в истории
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history = max_history
        self.generate_prompt = generate_prompt
        self.request_counter = 0
        
        # История сообщений для данной сессии
        self._history: List[Dict[str, str]] = []
        
        # Добавляем системный промпт
        self._add_to_history("system", system_prompt)
        
        # Инициализируем модель
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели LangChain"""
        self.chat_model = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def _add_to_history(self, role: str, content: str):
        """Добавляет сообщение в историю"""
        self._history.append({"role": role, "content": content})
        
        # Ограничиваем историю, сохраняя системный промпт
        if len(self._history) > self.max_history:
            system_message = self._history[0]
            recent_messages = self._history[-(self.max_history-1):]
            self._history = [system_message] + recent_messages
    
    def _history_to_messages(self) -> List[Any]:
        """Преобразует историю в формат LangChain сообщений"""
        messages = []
        for msg in self._history:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
    
    def send_message(
        self,
        message: str,
        use_history: bool = True
    ) -> str:
        """
        Отправляет сообщение модели и возвращает ответ.
        
        Args:
            message: текст сообщения пользователя
            use_history: использовать ли историю предыдущих сообщений
        
        Returns:
            ответ модели
        """
        # Добавляем сообщение пользователя в историю
        self._add_to_history("user", message)
        
        # Формируем сообщения для отправки
        if use_history:
            messages = self._history_to_messages()
        else:
            # Используем только системный промпт и текущее сообщение
            messages = [
                SystemMessage(content=self._history[0]["content"]),
                HumanMessage(content=message)
            ]
        
        # Отправляем запрос модели
        response = self.chat_model.invoke(messages)
     
        # Добавляем ответ ассистента в историю
        self._add_to_history("assistant", response.content)
        
        return response.content
    
    def get_history(self) -> List[Dict[str, str]]:
        """Возвращает полную историю текущей сессии"""
        return self._history.copy()
    

    def generate(self, x, y):
        TA1, RA1, _ = x
        TA2, RA2, _ = y
        response = self.chat_model.invoke(self.generate_prompt.format(TA1, RA1, TA2, RA2)).content
        self._add_to_history("llm", response)
        self.request_counter += 1
        return response
