import csv

class Logger:
    """Собирает статистики по эпизодам, пишет CSV и текстовые логи."""

    def __init__(self, stats_path: str, log_path: str | None = None) -> None:
        """Открывает/создаёт CSV файл, пишет заголовок."""
        ...

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        loss: float,
    ) -> None:
        """Добавляет строку в CSV и (опционально) печатает в консоль."""
        ...

    def get_dataframe(self) -> "pd.DataFrame":
        """Читает CSV и возвращает pandas DataFrame (для ноутбуков)."""
        ...

    def close(self) -> None:
        """Закрывает файловые дескрипторы."""
        ...