## Intro to Reinforcement Learning Example - by Genevieve Hayes ##
## Use Q-Learning to find the optimal policy for a simple maze environment ##

# Import libraries
import numpy as np
import mdp

# Create transition and reward matrices
def create_matrices(maze, reward, penalty_s, penalty_l, prob):
    """Create reward and transition matrices for input into the mdp QLearning
    function
    
    Args:
    maze: array. 0-1 numpy array giving the positions of the white cells
    (denoted 1) and the gray cells (denoted 0) in the maze;
    reward: float. Reward for reaching the end of the maze;
    penalty_s: float. Penalty for entering a white cell;
    penalty_l: float. Penalty for entering a gray cell;
    prob: float. Probability of moving in the intended direction.
    
    Returns:
    R: array. Reward matrix;
    T: array. Transition matrix.
    """
    
    r, c = np.shape(maze)
    states = r*c
    p = prob
    q = (1 - prob)*0.5
    
    # Create reward matrix
    path = maze*penalty_s
    walls = (1 - maze)*penalty_l
    combined = path + walls
    
    combined[-1, -1] = reward
            
    R = np.reshape(combined, states)
    
    # Create transition matrix
    T_up = np.zeros((states, states))
    T_left = np.zeros((states, states))
    T_right = np.zeros((states, states))
    T_down = np.zeros((states, states))
    
    wall_ind = np.where(R == penalty_l)[0]

    for i in range(states):
        # Up
        if (i - c) < 0 or (i - c) in wall_ind :
            T_up[i, i] += p
        else:
            T_up[i, i - c] += p
        
        if i%c == 0 or (i - 1) in wall_ind:
            T_up[i, i] += q
        else:
            T_up[i, i-1] += q
        
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_up[i, i] += q
        else:
            T_up[i, i+1] += q
            
        # Down
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_down[i, i] += p
        else:
            T_down[i, i + c] += p
        
        if i%c == 0 or (i - 1) in wall_ind:
            T_down[i, i] += q
        else:
            T_down[i, i-1] += q
        
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_down[i, i] += q
        else:
            T_down[i, i+1] += q
            
        # Left
        if i%c == 0 or (i - 1) in wall_ind:
            T_left[i, i] += p
        else:
            T_left[i, i-1] += p
            
        if (i - c) < 0 or (i - c) in wall_ind:
            T_left[i, i] += q
        else:
            T_left[i, i - c] += q
        
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_left[i, i] += q
        else:
            T_left[i, i + c] += q
        
        # Right
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_right[i, i] += p
        else:
            T_right[i, i+1] += p
            
        if (i - c) < 0 or (i - c) in wall_ind:
            T_right[i, i] += q
        else:
            T_right[i, i - c] += q
        
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_right[i, i] += q
        else:
            T_right[i, i + c] += q
    
    T = [T_up, T_left, T_right, T_down] 
    
    return T, R

# Define maze array
maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]]) 
    
# Create transition and reward matrices
T, R = create_matrices(maze, 1, -0.04, -0.75,  0.8)

# Set q-learning parameters
gamma = 0.9 # Discount factor
alpha = 0.3 # Learning rate
eps = 0.5 # Random action prob
decay = 1.0 # Decay rate
iters = 50000 # Number of iterations

# Run Q-learning algorithm to find optimal policy
np.random.seed(1)
q = mdp.QLearning(T, R, gamma, alpha, eps, decay, iters)
q.run()

# Print optimal policy
pol = np.reshape(np.array(list(q.policy)), np.shape(maze))
print(pol)

# Note: in output, 0 = Up, 1 = Left, 2 = Right, 3 = Down