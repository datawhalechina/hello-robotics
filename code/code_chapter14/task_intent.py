"""把长任务拆成简单动作意图，再交给普通π0.5输出连续动作。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskIntent:
    colors: tuple[str, ...] = ("red", "green", "blue")

    def next_color(self, completed: set[str]) -> str | None:
        return next((color for color in self.colors if color not in completed), None)

    def prompt(self, completed: set[str]) -> str:
        color = self.next_color(completed)
        if color is None:
            return "The task is complete. Keep the robot still."
        return f"Pick up the {color} block and place it into the empty box."

    def is_complete(self, completed: set[str]) -> bool:
        return all(color in completed for color in self.colors)

    @property
    def full_prompt(self) -> str:
        if len(self.colors) == 1:
            return f"Pick up the {self.colors[0]} block and place it into the empty box."
        return (
            "Pick up the red, green, and blue blocks one at a time and place "
            "all three blocks into the empty box on the table."
        )
