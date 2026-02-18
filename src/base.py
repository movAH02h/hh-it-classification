from abc import ABC, abstractmethod
from typing import Any, Optional


class ProcessingStep(ABC):
    """
    Базовый абстрактный класс для звеньев цепочки обработки данных.

    Реализует паттерн "Цепочка обязанностей" (Chain of Responsibility), позволяя
    последовательно применять различные этапы обработки к данным.

    Атрибуты:
        _next_step (ProcessingStep): Ссылка на следующий обработчик в цепочке.
    """

    def __init__(self) -> None:
        """Инициализирует обработчик без следующего звена."""
        self._next_step: Optional['ProcessingStep'] = None

    def set_next(self, step: 'ProcessingStep') -> 'ProcessingStep':
        """
        Устанавливает следующий обработчик в цепочке.

        Аргументы:
            step (ProcessingStep): Следующий обработчик в цепочке.

        Вернёт:
            ProcessingStep: Установленный обработчик для цепочного вызова.
        """
        self._next_step = step
        return step

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Обрабатывает данные и передает их следующему звену цепочки.

        Аргументы:
            data: Входные данные для обработки.

        Вернёт:
            Результат обработки последним звеном цепочки или исходные данные.
        """
        if self._next_step:
            return self._next_step.process(data)
        return data
