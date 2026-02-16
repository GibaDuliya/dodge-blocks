import csv
import os
import pandas as pd

class Logger:
    """Собирает статистики по эпизодам, пишет CSV и текстовые логи."""

    def __init__(self, stats_path: str, log_path: str | None = None) -> None:
        """Открывает/создаёт CSV файл, пишет заголовок."""
        self.stats_path = stats_path
        
        # Создаем папку, если её нет
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        # Создаем файл и пишем заголовок
        with open(self.stats_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "episode_length", "loss"])

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        loss: float,
    ) -> None:
        """Добавляет строку в CSV и печатает в консоль."""
        # Запись в файл
        with open(self.stats_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, episode_length, loss])
        
        # Вывод в консоль (каждые N раз можно фильтровать в Trainer, но здесь пишем всё)
        # Форматирование для красоты
        print(f"Ep: {episode:4d} | Reward: {total_reward:6.1f} | Steps: {episode_length:4d} | Loss: {loss:7.4f}")

    def get_dataframe(self) -> "pd.DataFrame":
        """Читает CSV и возвращает pandas DataFrame."""
        if os.path.exists(self.stats_path):
            return pd.read_csv(self.stats_path)
        return pd.DataFrame()

    def close(self) -> None:
        pass