This is the repository supplementing the Master Thesis "Explainable Reinforcement Learning for a Fully Autonomous Agent in Maritime Navigation". It contains an environment, a simple RL model and various explanation attempts involving LIME. LIME is supplied as a modified version, as the original one did have bugs.

## Setup

A virtual environment is recommended.

You can install the required packages with
```bash
python -m pip install -r requirements.txt
```

The program has been tested with Python 3.13.5 on Windows 10.

## Running the Simulation

Start the interactive simulation with:

```bash
python -m Controller.main
```

Under `Scenarios/`, you can select from several manually created scenarios.

### Controls

On the left side you can change visuals, controller settings and dt. The agent will use $dt = 5$ per default, ignoring the manually provided setting, if running using the Policy setting. Auto Run will execute a step frequently, with Manual mode respecting inputs, and both Policy modes using the model. Policy Stop will run until there has been a time-out ($step = 1000$) a collision with an obstacle or the goal has been reached.

### Input

On the right side, the input tab shows the Local View, Global View and the State Vector as received by the agent. The state vector allows for manually overwriting any values, which are then propagated as the new ground truth for the model input. Below that there is a break-down of the reward, with all but the total reward showing the reward from the current step.

### Interactive Environment (Env-Tools)

#### Important! Apply "Commit Changes and recalculate everything." before letting the agent run, after any changes.

Under Env-Tools you can create various polygons.

With Creation you can enter Creation mode, and then select corners of the boundary, before finalizing with a finish. An invalid edge, caused by a self cut is orange, and in the case of an invalid polygon, because the finishing edge would have a self cut, all edges are orange.

With Automatic Island, you can sample another island generated from the training distribution.

Generating an island will respect the Material Choice, with Void being able to be used to cut out arbitrary regions out of existing polygons. 

Selecting also respects polygon choice, allowing the three basic affine transformations to shape the polygon.

With the reset you have a quick way to save a certain scenario or ship state, to allow you to quickly iterate over similar starting conditions.

Lastly you can save and load scenarios, with `Scenarios/` containing a set of manually crafted scenarios and `Explanations/Samples/runXXXX` each containing a scenario each as well, for `run0000` to `run0072`

## Training the Agent

First, in order to not overwrite any existing models, set `instance_name = "SAC/07higherLR/run0"` under
`Agent/train.py`.

You may need to change additional settings, depending on your set-up, in particular if your GPU does not support cuda.

You can train a new model with
```bash
python -m Agent.train
```

## Training history

You can view all experiments with:
```bash
tensorboard --logdir=experiments
```

You can view SAC only experiments with:
```bash
tensorboard --logdir=experiments/SAC
```

You can view PPO only experiments with:
```bash
tensorboard --logdir=experiments/PPO
```

And lastly, to view the training history of the selected model only, you can use
```bash
tensorboard --logdir=experiments/SAC/07higherLR
```

## The Explanations
Under `Explanations\` you can find the results of applying modified LIME and SP-LIME versions to various scenarios. 

With `sample_gen.ipynb` the sampling was generated.

Under `Explanations\Basic\bias_checking.ipynb` you can find the statistics of a set of $10\,000$ start samples.

And `Explanations\LimeNotebooks` contains various examples of explanations, used in the thesis. Note that re-run results may differ, due to the way LIME samples random instances.
