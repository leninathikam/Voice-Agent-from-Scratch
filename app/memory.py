from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Memory:
    """Short conversation history sent with every model call."""

    system: str
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 20

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            self.turns = self.turns[overflow:]

    def as_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system}]
        messages.extend({"role": t.role, "content": t.content} for t in self.turns)
        return messages
