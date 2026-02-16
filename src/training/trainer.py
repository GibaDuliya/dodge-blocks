class Trainer:
    """Основной цикл обучения REINFORCE."""

    def __init__(
        self,
        env: "GameEnv",
        agent: "ReinforceAgent",
        train_config: "TrainConfig",
        logger: "Logger",
    ) -> None:
        """Сохраняет зависимости."""
        ...

    def train(self) -> dict:
        """
        Запускает цикл на num_episodes эпизодов.
        В каждом вызывает run_episode(), затем agent.update_policy().
        Периодически вызывает save_checkpoint() и logger.log_episode().
        Возвращает итоговый словарь статистик.
        """
        ...

    def run_episode(self) -> tuple[float, int]:
        """
        Один эпизод: reset env → цикл (select_action, step, store_reward)
        до done или max_steps. Возвращает (total_reward, steps).
        """
        ...

    def save_checkpoint(self, episode: int) -> None:
        """Вызывает agent.save() с путём из train_config.checkpoint_dir."""
        ...

    def evaluate(self, num_episodes: int) -> dict:
        """
        Запускает num_episodes без обучения (torch.no_grad),
        возвращает {'mean_reward': ..., 'mean_length': ..., 'std_reward': ...}.
        """
        ...
