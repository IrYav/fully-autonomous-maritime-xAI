from Agent.Curriculum.curriculum import Curriculum, Difficulty, GraduationCondition, ScenarioType

def build_curriculum():
  difficulties = [start_diff()]
  difficulties.append(_blocking_diff())
  difficulties.extend(_rest_diffs())
  difficulties.append(final_diff())
  return Curriculum(difficulties)

def start_diff() -> Difficulty:
  return Difficulty(0,ScenarioType.NONE,GraduationCondition(0.95),0)

def final_diff() -> Difficulty:
  return Difficulty(15,ScenarioType.NONE,GraduationCondition(1.1), 15)

def _blocking_diff() -> Difficulty:
  return Difficulty(1,ScenarioType.BLOCKING,GraduationCondition(0.9),1)

def _rest_diffs() -> list[Difficulty]:
  return [Difficulty(i, ScenarioType.NONE, GraduationCondition(0.8),i) for i in range(2,15)]
  
