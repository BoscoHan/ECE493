to run the training script for each algorithm change any parameters within the files themselves:

For example:
> "python3 training_scripts/train_trpo.py"


when training finishes, the zip file will be placed in the project directory with name < Algorithm_name >_slime_volleyball.zip

To use the newly trained model, move the generated zip file into < Algorithm_name >_slime_volleyball.zip into directory: zoo/< algorithm name > and rename the zip to *best_model.zip*

and then to evaluate the policy against other policies:
For example, running ppo against trpo:
> python3 eval_agents.py --left ppo --right trpo --render