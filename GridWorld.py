from collections import *
from glob import glob
from pickle import NONE
import copy
import random

# HW2 - Due 10/08

# Name: Jacob Schechter
# Course: CPSC 6420-001
# Date: 09/29/2022
# Environment: Python 3.9.12 ('tensor': conda)
################################################################################################################
#                                         # INSTRUCTIONS #                                                     #
################################################################################################################
# ~ In terminal/CL -  $ HW2_Jacob_Schechter.py                                                                 #
# ~ Will be given a prompt to select what problem to run (B -> F)                                              #
# ~ Can select by typing the respective letter (non-case sensitive) i.e. b                                     #
# This will print the first 10 iterations of the VI for the specified values (set by the chosen part)          #
# ~ Iff not part B.) it will also print the optimal path for ease of tracking values since I found it annoying #
# to come through each state via the method stated by part B                                                   #
# ~ Entering an invalid choice (i.e. not B -> F) ends the run and will need to be recompiled again             #
# ~ After printing the selected problem part, it will prompt to run another (type y) or quit (type n)          #
# ~ Running another will provide the same prompt first stated above                                            #
#                                                                                                              #
# Note: Printing Path seem to break, but works slowly                                                          # 
################################################################################################################


###### Grid Position #######
# State = (x,y,d)
# where x and y represent the horizontal and vertical positions (i.e. location),
# and d represents the direction the robot is facing (1: up, 2: down, 3: left, and 4: right). 

###### Robot Actions #######
# The robot can take the following actions: 
#  A1: Move one cell forward in the direction it is facing. Cost: 1.5 
#  A2:cells forward in the direction it is facing. Cost: 2 
#  A3: Turn to its left, and stay in the same cell. Cost: 0.5 
#  A4: Turn to its right, and stay in the same cell. Cost: 0.5 

###### Reward ######
# This can also be considered an immediate 
# negative reward. For example, we have R(s,A1,s’) = -1.5. The cost is evaluated on the current 
# state, (the state the robot is in when it begins the action, not the one it lands on after 
# performing the action). In the same way, the value of state V(s) represents the value of the 
# current state and you should initialize the algorithm with V1(5,5,x)=+100, V1(4,4,x)=-1000 (for 
# x=1,2,3,4 representing the robot orientation/direction), and zero for all other states.

###### Example: Possible state ######
# Example State (4,1,4) --> it means that it is in location (4,1) and facing right. 
# The result of possible actions for this state are as follows: 
# A1 (move 1 cell forward) --> (5,1,4) 
# A2 (move 2 cells forward) --> impossible  remains in the current state (4,1,4) 
# A3 (turn left) --> (4,1,1) : the robot stays in (4,1) but now faces up 
# A4 (turn right) --> (4,1,2) : the robot stays in (4,1) but now faces down

###### Impossible States ######
# A move is impossible if it would result in landing on a blocked cell, like (2,2), (2,3), or (3,2). 
# Or if it would result in crossing a barrier, like moving from state (2,5) to (3,5), or (5,3) to (5,4). 
# A move that would take the robot outside of our 5x5 grid is also impossible.


###### Initializing Components ######
global NUM_ROWS
global NUM_COLS
global ACTIONS
global START_POS
global GAMMA 
global NOISE 
global LIVING_REWARD



NUM_ROWS = 5
NUM_COLS = 5
GOAL = [5,5]
GAME_OVER = [4,4]
START_POS = [1,1,4] #x,y,d

GAMMA = 1.0
NOISE = 0.0
LIVING_REWARD = 0.0

# refer back to VI pseudocode Lecture 06-slide 39
global prev_state # k vector
global curr_state # k+1 vector



ACTIONS = [["A1",1.5], ["A2",2], ["A3",0.5], ["A4",0.5]]

vert_walls = [[2,5]] #represent the walls '|' between columns i.e. a wall at (2,5) places a wall between column 2 and 3
horiz_walls = [[5,3]] #represent the walls '__' between rows i.e. a wall at (5,3) places a wall between row 3 and 4 at(impassable direction)
invalid_pos = [[2,2],[2,3],[3,2]] #invalid state positions in grid
num_iterations = 100

class State:
   def __init__(self,invalid,x,y,d,invalid_act,value,isGoal,isGmOver,bestAction) -> None:
      self.invalid = invalid
      self.x = x
      self.y = y
      self.d = d
      self.invalid_act = invalid_act # list of invalid actions in current direction built through robot movement
      self.value = value
      self.isGoal = isGoal
      self.isGmOver = isGmOver
      self.bestAction = bestAction


# Initialze VI grid, k = 0 iteration
def build_grid(rows,cols,action):
   curr_state = [] #intializing vector of all states at iteration k = 0
   
   # Obtaining all states 
   for x in range(1,cols+1):
      for y in range(1,rows+1):
         for d in range(1,len(action)+1):
            invalid = 0
            for pos in invalid_pos:
               if (x == pos[0] and y == pos[1]):
                  invalid = 1
            if (invalid):
                  curr_state.append(State(True,x,y,d,[],0,False,False,None))
            else:
               curr_state.append(State(False,x,y,d,[],0.0,False,False,None))
            
   return curr_state



def transition_model(curr_state,s,a1):
   # No chosen action
   cost = 0
   temp_x = s.x
   temp_y = s.y
   temp_d = s.d

   # Move forward 1 cell
   if a1==ACTIONS[0][0]:
      cost = ACTIONS[0][1]
      if temp_d==1: # UP
         temp_y+=1
      elif temp_d==2: # DOWN
         temp_y-=1
      elif temp_d==3: # LEFT
         temp_x-=1
      elif temp_d==4: # RIGHT
         temp_x+=1
      else:
         print("error, move 1 invalid direction")
   # Move forward 2 cells
   elif a1==ACTIONS[1][0]:
      cost = ACTIONS[1][1]
      if temp_d==1: # UP
         temp_y+=2
      elif temp_d==2: # DOWN
         temp_y-=2
      elif temp_d==3: # LEFT
         temp_x-=2
      elif temp_d==4: # RIGHT
         temp_x+=2
      else:
         print("error, move 2 invalid direction")
   # Turn left in place
   elif a1==ACTIONS[2][0]:
      cost = ACTIONS[2][1]
      if temp_d==1: # UP->LEFT
         temp_d=3
      elif temp_d==2: # DOWN->RIGHT
         temp_d=4
      elif temp_d==3: # LEFT->DOWN
         temp_d=2
      elif temp_d==4: # RIGHT->UP
         temp_d=1
      else:
         print("error, left rot. invalid direction")
   # Turn right in place
   elif a1==ACTIONS[3][0]:
      cost = ACTIONS[3][1]
      if temp_d==1: # UP->RIGHT
         temp_d=4
      elif temp_d==2: # DOWN->LEFT
         temp_d=3
      elif temp_d==3: # LEFT->UP
         temp_d=1
      elif temp_d==4: # RIGHT->DOWN
         temp_d=2
      else:
         print("error, right rot. invalid direction")
   else:
      print("error, direction incorrect")
      exit(-1)

   next_state = None
   viable = True
   for iter in curr_state:
      if temp_x==iter.x and temp_y==iter.y and temp_d==iter.d:
         next_state = iter
         break

   
   if next_state != None:
      for v in vert_walls:
         if ((s.x<=v[0] and v[0]<next_state.x) or (next_state.x<=v[0] and v[0]<s.x)) and (next_state.y==s.y and next_state.y==v[1]):
            viable = False
            if a1=="A2":
               # print("vertical")
               break
      if viable:
         for h in horiz_walls:
            if ((s.y<=h[1] and h[1]<next_state.y) or (next_state.y<=h[1] and h[1]<s.y)) and (next_state.x==s.x and next_state.x==h[0]):
               viable = False
               break
      if viable:
         for emp in invalid_pos:
            if ((s.x<=emp[0] and emp[0]<next_state.x) or (next_state.x<=emp[0] and emp[0]<s.x)) and (next_state.y==s.y and next_state.y==emp[1]):
               viable=False
               break
            if ((s.y<=emp[1] and emp[1]<next_state.y) or (next_state.y<=emp[1] and emp[1]<s.y)) and (next_state.x==s.x and next_state.x==emp[0]):
               viable=False
               break
   else:
      # Next state does not exist
       next_state = s
       viable = False
       cost = 0
   return next_state,viable,cost


# VI w/ NOISE calculation
def rnd_Action(rnd_states,s,num_act):
   randi_best = copy.deepcopy(s)
   upd_Vof_s = 0
   for ps in rnd_states:
      if ps[0].x==s[0].x and ps[0].y==s[0].y and ps[0].d==s[0].d:
         upd_Vof_s += (1-NOISE) * ps[2]
      else:
         upd_Vof_s += ps[2] * (NOISE/num_act)
   
   randi_best[2] = upd_Vof_s
   return randi_best



# Remember assume immediate negative rewards, refer back to explanation sheet Dr. Razi drew out
def opt_state(curr_state,s):
   # Possible new states
   results = []
   Vof_s = 0

   # V(s) Calculation for best action
   for a1 in ACTIONS:
      next_state, viable, cost = transition_model(curr_state,s,a1[0])
      if viable:
         Vof_s = (GAMMA * next_state.value)-cost
         temp = [next_state,cost,Vof_s,a1,s]

         results.append(temp)

   ###### implement random action probability

   result_states = []
   if NOISE>0.0:
      iter_rnd = copy.deepcopy(results)
      # iterate through V(s) = P(s,a,s')*Vof_s
      for iter in iter_rnd:
         result_states.append(rnd_Action(results,iter,len(results)-1))
   else:
      result_states = copy.deepcopy(results)


   temp_best = []
   for result in result_states:
      if len(temp_best)==0:
         temp_best = copy.deepcopy(result)
         act = result[3][0]
      elif temp_best[2]<result[2]:
         temp_best = copy.deepcopy(result)
         act = result[3][0]
      elif temp_best[2]==result[2]:
         randi = round(random.uniform(0,1))
         if randi==1:
            act = result[3][0]
            temp_best = copy.deepcopy(result)

   return temp_best


# temporary placement
# # Initialize vector of states beginning from bottom left to top right
# curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)


# Value Iteration implementation
def value_iter(curr_state,horizon):
   # k = 1 -> Assign/Identify Goal, Gameover, invalid pos., and immediate 
   for k in range(1,horizon*5,1):
      prev_state = copy.deepcopy(curr_state)
      if k==1:
         # Special state case check
         for s in curr_state:
            if (s.x==GOAL[0] and s.y==GOAL[1]):
               s.value = 100.0
               s.isGoal = True
            elif (s.x==GAME_OVER[0] and s.y==GAME_OVER[1]):
               s.value = -1000.0
               s.isGmOver = True
            elif s.invalid:
               s.isGoal = False
               s.isGmOver = False
            else:
               temp_store = opt_state(prev_state,s)
               if s.x==temp_store[4].x and s.y==temp_store[4].y and s.d==temp_store[4].d: 
                  s.value = temp_store[2]
                  s.bestAction = temp_store[3][0]
      else:  

         for s in curr_state:
            # check if Goal, GameOver, or Blacked out grid position
            if (s.isGoal or s.isGmOver or s.invalid):
               continue
            else:
               temp = opt_state(prev_state,s)
               if s.x==temp[4].x and s.y==temp[4].y and s.d==temp[4].d:
                  s.value = temp[2]
                  s.bestAction = temp[3][0]
      
      if k<=10 or k==horizon*5-1:
         print("\niter "+str(k)+":")
         for j in curr_state:
            if j.isGoal:
               print("state (",str(j.x),",",str(j.y),",",str(j.d),")  V = (",str(j.value),")  Best Action = ",str(j.bestAction))
            elif j.isGmOver:
               print("state (",str(j.x),",",str(j.y),",",str(j.d),")  V = (",str(j.value),")  Best Action = ",str(j.bestAction))
            elif not j.invalid:
               print("state (",str(j.x),",",str(j.y),",",str(j.d),")  V = (",str(j.value),")  Best Action = ",str(j.bestAction))
         print("-----------------------------------------------")
   return curr_state
      
def get_opt_path(curr_state,horizon):
   print("----------------Optimal Pathing----------------")
   print("-----------------------------------------------")
   path_x = START_POS[0] # x
   path_y = START_POS[1] # y
   path_d = START_POS[2] # d
   optimal_pathing = []
   check_values = []
   found = 0

   # Attempt to find optimal path
   while True:
      view_state = copy.deepcopy(curr_state)
      for s in view_state:
         if s.x==path_x and s.y==path_y and s.d==path_d:
            add = [s.x,s.y,s.d]
            print("(x,y,d) => ("+str(s.x)+","+str(s.y)+","+str(s.d)+")")
            optimal_pathing.append(s)
            if s.bestAction=="A1":
               if s.d==1: # Up 1 cell
                  path_y+=1
                  print("  Action Taken: (A1) Up 1")
               elif s.d==2: # Down 1 cell
                  path_y-=1
                  print("  Action Taken: (A1) Down 1")
               elif s.d==3: # Left 1 cell
                  path_x-=1
                  print("  Action Taken: (A1) Left 1")
               elif s.d==4: # Right 1 cell
                  path_x+=1
                  print("  Action Taken: (A1) Right 1")
            elif s.bestAction=="A2":
               if s.d==1: # Up 2 cells
                  path_y+=2
                  print("  Action Taken: (A2) Up 2")
               elif s.d==2: # Down 2 cells
                  path_y-=2
                  print("  Action Taken: (A2) Down 2")
               elif s.d==3: # Left 2 cells
                  path_x-=2
                  print("  Action Taken: (A2) Left 2")
               elif s.d==4: # Right 2 cells
                  path_x+=2
                  print("  Action Taken: (A2) Right 2")
            elif s.bestAction=="A3":
               if path_d==1: # UP->LEFT
                  path_d=3
                  print("  Action Taken: (A3) Turn Left (Left)")
               elif path_d==2: # DOWN->RIGHT
                  path_d=4
                  print("  Action Taken: (A3) Turn Left (Right)")
               elif path_d==3: # LEFT->DOWN
                  path_d=2
                  print("  Action Taken: (A3) Turn Left (Down)")
               elif path_d==4: # RIGHT->UP
                  path_d=1
                  print("  Action Taken: (A3) Turn Left (Up)")
            elif s.bestAction=="A4":
               if path_d==1: # UP->RIGHT
                  path_d=4
                  print("  Action Taken: (A4) Turn Right (Right)")
               elif path_d==2: # DOWN->LEFT
                  path_d=3
                  print("  Action Taken: (A4) Turn Right (Left)")
               elif path_d==3: # LEFT->UP
                  path_d=1
                  print("  Action Taken: (A4) Turn Right (Up)")
               elif path_d==4: # RIGHT->DOWN
                  path_d=2
                  print("  Action Taken: (A4) Turn Right (Down)")
            elif s.bestAction==None:
               if s.x==GOAL[0] and s.y==GOAL[1]:
                  print("Reached Goal!")
                  return False
               elif s.x==GAME_OVER[0] and s.y==GAME_OVER[1]:
                  print("Failed - GAME OVER")
                  return False
            else:
               print("Broke out somehow")
            if add in check_values:
               print("Unable to reach Goal")
               return False
            else:
               check_values.append(add)
      # Pathing continues (i.e. found)
      if len(optimal_pathing)==found:
         print("Optimal Path failed")
         return None
      else:
         found = len(optimal_pathing)

         

# A.) If there is no living reward/penalty, no noise, and no discount (gamma = 1), use your
#     common sense to find the best possible route from (1,1) to (5,5)

   # Answer located in HW2_Jacob_Schechter.txt

# B.) With no discount (gamma = 1), no living reward, and no noise, use the Value Iteration
#     Algorithm with 100 iterations to update the optimal values for each state and print the
#     result [only for the first 10 iterations] in the following format:
   # Pasted in hw2.txt (could differ from live prints due to random lowest tied cost)

def main():
   running = True
   global GAMMA
   global NOISE

   while running:
      run = input((('Which problem do you want to run?'+
      '\n- Type B/b for part B (gamma = 1, living reward = 0, noise = 0)'+
      '\n- Type C/c for part C (start_state = (1,1,4), gamma = 1, living reward = 0, noise = 0)'+
      '\n- Type D/d for part D (start_state = (1,1,4), gamma = 0.8, living reward = 0, noise = 0)'+
      '\n- Type E/e for part E (start_state = (1,1,4), gamma = 0.2, living reward = 0, noise = 0)'+
      '\n- Type F/f for part F (start_state = (1,1,4), gamma = 0.9, living reward = 0, noise = 0.2)\n')))

      if run.lower()=='b':
         GAMMA = 1
         LIVING_REWARD = 0
         NOISE = 0
         print("Running B.) VI -> gamma = "+str(GAMMA)+", living reward = "+str(LIVING_REWARD)+", noise = "+str(NOISE))
         # Create vector of all states (k = 0)
         curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)
         # k = 1,...,H
         value_iter(curr_state,num_iterations)
      elif run.lower()=='c':
         GAMMA = 1
         LIVING_REWARD = 0
         NOISE = 0
         START_POS = [1,1,4] # x, y, d
         print("Running C.) VI and Optimal Path -> gamma = "+str(GAMMA)+", living reward = "+str(LIVING_REWARD)+", noise = "+str(NOISE))
         # Create vector of all states (k = 0)
         curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)
         # k = 1,...,H
         value_iter(curr_state,num_iterations)
         get_opt_path(curr_state,num_iterations)
      elif run.lower()=='d':
         GAMMA = 0.8
         LIVING_REWARD = 0
         NOISE = 0
         START_POS = [1,1,4] # x, y, d
         print("Running D.) VI -> gamma = "+str(GAMMA)+", living reward = "+str(LIVING_REWARD)+", noise = "+str(NOISE))
         # Create vector of all states (k = 0)
         curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)
         # k = 1,...,H
         value_iter(curr_state,num_iterations)
         get_opt_path(curr_state,num_iterations)
      elif run.lower()=='e':
         GAMMA = 0.2
         LIVING_REWARD = 0
         NOISE = 0
         START_POS = [1,1,4] # x, y, d
         print("Running E.) VI -> gamma = "+str(GAMMA)+", living reward = "+str(LIVING_REWARD)+", noise = "+str(NOISE))
         # Create vector of all states (k = 0)
         curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)
         # k = 1,...,H
         value_iter(curr_state,num_iterations)
         get_opt_path(curr_state,num_iterations)
      elif run.lower()=='f':
         GAMMA = 0.9
         LIVING_REWARD = 0
         NOISE = 0.2
         START_POS = [1,1,4] # x, y, d
         print("Running F.) VI -> gamma = "+str(GAMMA)+", living reward = "+str(LIVING_REWARD)+", noise = "+str(NOISE))
         # Create vector of all states (k = 0)
         curr_state = build_grid(NUM_ROWS,NUM_COLS,ACTIONS)
         # k = 1,...,H
         value_iter(curr_state,num_iterations)
         get_opt_path(curr_state,num_iterations)
      else:
         print("----------------------------------------------\nInvalid choice: Program broke - run main again\n----------------------------------------------\n")
         exit(-1)
         
      again = input("\nEnter Y/y to continue or N/n to stop\n")
      while (again.lower()!='y' and again.lower()!='n'):
         again = input("Invalid Choice: Enter Y/y to continue or N/n to stop\n")

      if again.lower()=='n':
            running = False
   


if __name__=="__main__":
   main()