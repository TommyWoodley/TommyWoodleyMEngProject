# Repository

## Running the Python Script

To run the `main.py` file for reinforcement learning training for tethered drone perching, you need to provide several arguments to specify the configuration of the training process. Below is a description of each argument and examples of how to use them.

### Arguments

- `-t` or `--timesteps`: **Required**. The number of timesteps for training. For example, `40000`.
- `-algo` or `--algorithm`: **Required**. The choice of algorithm. Options are `SAC` or `SACfD`.
- `-o` or `--output-filename`: **Optional**. The filename for storing logs. If not provided; logs, and graphs will NOT be generated. The filename provided will be appended with a timestamp. This will NOT overwritte previous created logs/files.
- `--save-replay-buffer`: **Optional**. If specified, the replay buffer will also be saved along with the model. NOTE: Replay buffers can be quite large in size - 100-200MB so regularly saving replay buffers can be an expensive operation.
- `-gui`: **Optional**. If specified, enables the graphical user interface for the environment. This is significantly slower for training purposes (around 5-10x slower). Therefore this shouldn't be used for training 
- `--demo-path`: **Optional**. The path to demonstration files. Default is `/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/Data/PreviousWorkTrajectories/rl_demos`. This is unlikely to be suitable for any other system so will need to be set.
- `--show-demo`: **Optional**. If specified, shows the demonstrations in the visual environment. Demonstrations will only show 
- `--checkpoint-episodes`: **Optional**. Frequency of checkpoint episodes. Default is `5000`.
- `--no-checkpoint`: **Optional**. If specified, this will overwrite the checkpoint epiosde value and perform no checkpointing during training. This can speed up training. Each checkpoint will take roughly 30 seconds so removing checkpointing can save some time during training.
- `-params` or `--hyperparams`: **Optional**. Overwrite hyperparameters by providing key-value pairs. For example, `lr:0.01 batch_size:10`.
- `-i` or `--trained-agent`: **Optional**. Path to a pretrained agent to continue training. This should be the full path to a `.zip` file containing the pre-trained agent. NOTE: Replay Buffers may not be carried over from the pre-trained agent. Buffer size will be stated prior to training to check this.

### Example Usage

#### Basic Example
To run the training process for 4000 timesteps, using the Soft Actor Critic from Demonstrations Algorithm with an output file log.
```sh
python main.py -t 40000 -algo SACfD -o output
```
