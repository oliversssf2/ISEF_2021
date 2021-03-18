# Plan

## DPLM class

The purpose of this class is to create a DPLM object that:

1. Initializes a DPLM with given parameters.

2. Performs all the calculations (moment, position, angle, etc.) and returns the requested variables of the DPLM When its functions are called.

3. Store the variables of the DPLM object in different states for reusing when its functions are called again with the exact same parameters.

## DPLM environment

### DPLM configuration

- n springs
- l slots for spring installation on each linkage (in total $2n-1$ ways to install a single spring)

- 