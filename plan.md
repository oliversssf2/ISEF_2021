# Plan

## DPLM class

The purpose of this class is to create a DPLM object that:

1. Initializes a DPLM with given parameters.

2. Performs all the calculations (moment, position, angle, etc.) and returns the requested variables of the DPLM When its functions are called.

3. Store the variables of the DPLM object in different states for reusing when its functions are called again with the exact same parameters.

### Function required:

1. A change state function that takes in the change of slot of each springs as input).
2. A return state function
3. A RMSE calculation function (used as the reward for the action taken).
4. Internal calculation functions.

## DPLM environment

### DPLM configuration

- n springs
- l slots for spring installation on each linkage (in total $2l-1$ ways to install a single spring)

- each of the three springs can either remain in the original slot (+0), move to the left r right adjacent slot (+1/-1), move to the third slot to its either side (+3/-3). So there are in total 5^n possible actions. (Tuple(5,5,...,5)(n in total))
  
### init

setup the followings:

- action space: Tuple (n*Discrete(5))
- observation space: Tuple (n*Discrete($2l-1$))
- initiatiate the DPLM with a random state
- set maximum episode length (max step)

### step

1. change state by calling the DPLM object (spring installation position) according to action taken by the algorithm.
2. Give reward according to updated state (RMSE across all angles)
3. check if the mission is done (probably RMSE within certain range or if maximum step reached)
4. return info

### render

Just print some values probably.

### reset

Assign a random state for the DPLM

## The main file

Basically a while loop with the algorithm and some stopping conditions.
