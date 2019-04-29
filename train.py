import argparse                                         # Allows parsing arguments in the command line
import os                                               # os handles directory/workspace changes
from random import random, randint, sample              # Handles random operations
from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters and do version control
from datetime import datetime                           # datetime to use proper date and time formats
import numpy as np                                      # NumPy to handle numeric and NaN operations
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn                                    # nn for neural network layers
from src.deep_q_network import DeepQNetwork             # Deep Q Network model
from src.flappy_bird import FlappyBird                  # Flappy Compass game interface
from src.utils import pre_processing                    # Image pre-processing method


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="Optimization algorithm.")
    parser.add_argument("--lr", type=float, default=1e-6, help="The learning rate applied in the training process.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reinforcement learning policy parameter that influences non-terminal reward values.")
    parser.add_argument("--initial_epsilon", type=float, default=0.1, help="Initial epsilon value that indicates the initial probability of random actions.")
    parser.add_argument("--final_epsilon", type=float, default=1e-4, help="Final epsilon value that indicates the final probability of random actions.")
    parser.add_argument("--num_iters", type=int, default=2000000, help="Number of training iterations.")
    parser.add_argument("--iters_to_save", type=int, default=1000000, help="Iterations intervals when the model is saved")
    parser.add_argument("--replay_memory_size", type=int, default=50000, help="Number of epoches between testing phases")
    parser.add_argument("--conv_dim", type=int, nargs='+', help="Number of filters, i.e. filter depth, of each convolutional layer.")
    parser.add_argument("--conv_kernel_sizes", nargs='+', type=int, help="Kernel size (or convolutional matrix dimension) of each convolutional layer.")
    parser.add_argument("--conv_strides", nargs='+', type=int, help="Stride used in each convolutional layer.")
    parser.add_argument("--fc_dim", type=int, nargs='*', help="Output dimension of each hidden fully connected layer (if there are any).")
    parser.add_argument("--random_seed", type=int, default=123, help="Seed value used in random operations. Using the same value allows reproducibility.")
    parser.add_argument("--saved_path", type=str, default="models", help="Directory where trained models will be saved.")
    parser.add_argument("--log_comet_ml", type=bool, default=False, help="Indicates whether to save training stats on Comet.ml.")
    parser.add_argument("--comet_ml_api_key", type=str, default="", help="API key to be able to log data to Comet.ml.")
    parser.add_argument("--comet_ml_project_name", type=str, default="", help="Comet.ml project name.")
    parser.add_argument("--comet_ml_workspace", type=str, default="", help="Comet.ml workspace.")
    parser.add_argument("--comet_ml_save_model", type=bool, default=False, help="If true, models are also uploaded to Comet.ml.")

    args = parser.parse_args()
    return args


def train(opt):
    # Set random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.random_seed)
    else:
        torch.manual_seed(opt.random_seed)
    # Instantiate the model
    if opt.conv_dim is not None and \
       opt.conv_kernel_sizes is not None and \
       opt.conv_strides is not None and \
       opt.fc_dim is not None:
        model = DeepQNetwork(opt.image_size, opt.image_size, conv_dim=opt.conv_dim, conv_kernel_sizes=opt.conv_kernel_sizes, 
                             conv_strides=opt.conv_strides, fc_dim=opt.fc_dim)
    else:
        model = DeepQNetwork(opt.image_size, opt.image_size)

    if opt.log_comet_ml:
        # Create a Comet.ml experiment
        experiment = Experiment(api_key=opt.comet_ml_api_key,
                                project_name=opt.comet_ml_project_name, workspace=opt.comet_ml_workspace)
        experiment.log_other("iters_to_save", opt.iters_to_save)
        experiment.log_other("completed", False)
        experiment.log_other("random_seed", opt.random_seed)

        # Report hyperparameters to Comet.ml
        hyper_params = {"image_size": opt.image_size,
                        "batch_size": opt.batch_size,
                        "optimizer": opt.optimizer,
                        "learning_rate": opt.lr,
                        "gamma": opt.gamma,
                        "initial_epsilon": opt.initial_epsilon,
                        "final_epsilon": opt.final_epsilon,
                        "num_iters": opt.num_iters,
                        "replay_memory_size": opt.replay_memory_size,
                        "random_seed": opt.random_seed,
                        "conv_dim": opt.conv_dim,
                        "conv_kernel_sizes": opt.conv_kernel_sizes,
                        "conv_strides": opt.conv_strides,
                        "fc_dim": opt.fc_dim}
        experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)   # Optimization algorithm
    criterion = nn.MSELoss()                                    # Loss function
    game_state = FlappyBird()                                   # Instantiate the Flappy Compass game
    image, reward, terminal = game_state.next_frame(0)          # Get the next image, along with its reward and an indication if it's a terminal state

    # Image preprocessing step (scaling, color removal and convertion to a PyTorch tensor)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)

    # Move the model and the current image data to the GPU, if available
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()

    # Prepare the state variable, which will host the last 4 frames
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    # Initialize the replay memory, which saves sets of consecutive game states, the reward and terminal state indicator
    # so that the model can learn from them (essentially constitutes the training data, which grows with every new iteration)
    replay_memory = []
    
    iter = 0                                                    # Iteration counter

    # Main training loop performing the number of iterations specified by num_iters
    while iter < opt.num_iters:
        prediction = model(state)[0]                            # Get a prediction from the current state
        epsilon = opt.final_epsilon + (
                  (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters) # Set the decay of the probability of random actions
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:
            # Use the model's prediction to decide the next action
            action = torch.argmax(prediction).item()

        # Get a new frame and process it
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
        next_image = torch.from_numpy(next_image)

        # Move the next image data to the GPU, if available
        if torch.cuda.is_available():
            next_image = next_image.cuda()

        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]     # Prepare the next state variable, which will host the last 4 frames
        replay_memory.append([state, action, reward, next_state, terminal])         # Save the current state, action, next state and terminal state indicator in the replay memory
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]                                                    # Delete the oldest reolay from memory if full capacity has been reached
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))      # Retrieve past play sequences from the replay memory
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))              # States of the current batch
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32)) # Actions taken in the current batch
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])  # Rewards in the current batch
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))    # Next states of the current batch

        # Move batch data to the GPU, if available
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)           # Predictions of the model for the replays of the current batch
        next_prediction_batch = model(next_state_batch)         # Next predictions of the model for the replays of the current batch

        # Set ground truth for the rewards for the current batch, considering whether the state is terminal or not
        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1) # Predicted Q values (i.e. estimated return for each action)
        optimizer.zero_grad()                                   # Reset the gradients to zero before a new optimization step
        loss = criterion(q_value, y_batch)                      # Calculate the loss
        loss.backward()                                         # Backpropagation
        optimizer.step()                                        # Weights optimization step

        state = next_state                                      # Move to the next frame
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        
        if opt.log_comet_ml:
            # Log metrics to Comet.ml
            experiment.log_metric("train_loss", loss, step=iter)
            experiment.log_metric("train_epsilon", epsilon, step=iter)
            experiment.log_metric("train_reward", reward, step=iter)
            experiment.log_metric("train_Q_value", torch.max(prediction), step=iter)

        if (iter+1) % opt.iters_to_save == 0:
            # Get the current day and time to attach to the saved model's name
            current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

            # Set saved model name
            model_filename = f'{opt.saved_path}/flappy_compass_{current_datetime}_{iter+1}.pth'

            # Save model every iters_to_save iterations
            torch.save(model, model_filename)

            if opt.log_comet_ml and opt.comet_ml_save_model:
                # Upload model to Comet.ml
                experiment.log_asset(file_path=model_filename, overwrite=True)

    # Get the current day and time to attach to the saved model's name
    current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

    # Set saved model name
    model_filename = f'{opt.saved_path}/flappy_compass_{current_datetime}_{iter+1}.pth'

    # Save the model after reaching the final iteration
    torch.save(model, model_filename)

    if opt.log_comet_ml:
        # Only report that the experiment completed successfully if it finished the training without errors
        experiment.log_other("completed", True)

        if opt.comet_ml_save_model:
            # Upload model to Comet.ml
            experiment.log_asset(file_path=model_filename, overwrite=True)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
