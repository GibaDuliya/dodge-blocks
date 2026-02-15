import pygame

class GameRenderer:
    """Pygame-визуализация состояния GameEnv."""

    def __init__(self, env_config: "EnvConfig", render_config: "RenderConfig") -> None:
        """Сохраняет конфиги, НЕ инициализирует дисплей (ленивая инициализация)."""
        ...

    def init_display(self) -> None:
        """pygame.init(), создаёт окно нужного размера."""
        ...

    def render(self, state: "np.ndarray", score: int) -> None:
        """
        Отрисовывает один кадр:
        - фон / сетку
        - агента
        - падающий блок
        - текст со счётом
        Вызывает pygame.display.flip().
        """
        ...

    def close(self) -> None:
        """pygame.quit()."""
        ...

    @staticmethod
    def handle_events() -> bool:
        """
        Обрабатывает очередь событий pygame.
        Возвращает False если пользователь закрыл окно.
        """
        ...