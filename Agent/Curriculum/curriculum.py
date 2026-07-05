from dataclasses import dataclass
from enum import Enum, auto


class ScenarioType(Enum):
  NONE = auto()
  BLOCKING = auto()
  CANYON = auto()

@dataclass(frozen=True)
class GraduationCondition:
  min_success_rate: float
  def is_met(self, success_rate: float) -> bool:
    return success_rate >= self.min_success_rate


@dataclass(frozen=True)
class Difficulty:
  num_islands: int
  scenario: ScenarioType
  graduation: GraduationCondition
  index: int #Currently equal to num_islands, but might deviate later.

@dataclass(frozen=True)
class Curriculum:
  difficulties: list[Difficulty]

  def passed(self, current: Difficulty, success_rate: float) -> bool:
    return (
      current.graduation.is_met(success_rate) and 
      current.index < len(self.difficulties) - 1
    )

  def next_difficulty(self, current : Difficulty) -> Difficulty:
    return self.difficulties[current.index + 1]

  def get_start_diff(self) -> Difficulty:
    return self.difficulties[0]