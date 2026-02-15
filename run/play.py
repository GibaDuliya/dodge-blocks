"""
Интерактивный режим:
  --mode human   → управление стрелками (для отладки среды)
  --mode agent   → загрузить чекпоинт и наблюдать за агентом
Оба режима используют GameRenderer для визуализации.
"""

def parse_args() -> "argparse.Namespace":
    ...

def play_human(env, renderer) -> None:
    ...

def play_agent(env, agent, renderer) -> None:
    ...

def main() -> None:
    ...

if __name__ == "__main__":
    main()