from PySide6.QtWidgets import (
  QWidget
)

class LimeExplTab(QWidget):
  #There is always exactly (or at most?) one metric chosen.
  active_metric = None # Probably a categorical, maybe enum.
  #We will probably need something form the matplotlib (e.g., to render GT):
  plot_ax = None #Feed via init

  #An explanation here is always 
  # a mapping from an observation (12D) 
  # to one of the following singular instances:

  #I want:
  #LIME Explanations:
  #LIME explaining a step: (SUBTAB)
    #local reward <- ac.model_step(True)["reward"]
    #rudder <- ac.model_step(True)["action"][0]
    #velocity <- ac.model_step(True)["action"][1]

  #LIME explaining a run: (SUBTAB):
    #Reward Delta <- Small auxilary, ac.finish_episode(True)[0] - base_reward
    #Step Count to goal <- ac.finish_episode(True)[1]
    #Collisions (Categorical) <- ac.finish_episode(True)[2] but modified, 
      # Four! categories. normal 3, plus time-out ()

  #Ground truth checks (one for each subtab):
    #Pos x, Pos y, heading -> selected explanation metric.
    #Show only one, but calculate them in parallel (Button to trigger calc).
    #Technically a set of explanations, as they map 12D to a singular instance,
    # but only 3 sizes get perturbed, rest stays the same
  
  #-----------------
  #Intended Workflow, we want to extract data-flow insights from this:
  #We only assume step-explanations for now. Run explanations are fully analogous.
  #Two important buttons:
  #Calc LIME <- Now we have a lime explanation for the given state.
    # The explanation is not made for other states, but nonetheless has 
    # a defined value which can be evaluated and compared.
      #This is intended to act as a metric of "necessary locality"
  #Calc GT  <- Now we *many* samples over a ground truth.
    #Some kind of distribution, domain should be (X,Y, Heading).
    #For each d in distribution, we can pick a few choices n from each:
      #LIME (maybe calc on demand)
      #GT (definitely store)
      #GT minus LIME (possibly some kind of weighting).
    #Therefor, we have a 3,n matrix where one can select exactly one thing.
