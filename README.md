# Robot occupying game
This project aims to implement a RL agent to play this game described below:
## Game Description
There are 2 robots occupying the blocks. Each time we have 5 blocks to be selected.
Black and yellow blocks are seen as obstacles. If robots hits the black area will get the score -10 while yellow -4. The score will also -10 when the 2 robots hit each other.
Only when the robot choose the free block and no collision happening, scores plus 8.
The initial position of the robot is shown in the figure.
<img width="831" alt="截屏2024-04-17 15 39 48" src="https://github.com/Ma77-Tasuga/DQN_test_on_robotgame/assets/165873848/cbe2f2b7-5ace-416e-b874-ff292c714bd5">
