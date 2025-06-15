//
//  tutorial.cpp
//  RLTutorial
//
//  Created by Julio Godoy on 11/25/18.
//  Copyright © 2018 Julio Godoy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2,current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward,Qvalues[100][100][4], reward[100][100],finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos,y_pos, prev_x_pos, prev_y_pos, blockedx, blockedy,i,j,k;
ofstream reward_output;

//////////////
//Setting value for learning parameters
int action_sel=2; // 1 is greedy, 2 is e-greedy
int environment= 1; // 1 is small grid, 2 is Cliff walking
int algorithm = 1; //1 is Q-learning, 2 is Sarsa
int stochastic_actions=0; // 0 is deterministic actions, 1 for stochastic actions
int num_episodes=3000; //total learning episodes
float learn_rate=0.1; // how much the agent weights each new sample
float disc_factor=0.99; // how much the agent weights future rewards
float exp_rate=0.3; // how much the agent explores
///////////////


void Initialize_environment()
{
    if(environment==1)
    {
        
        height_grid= 3;
        width_grid=4;
        goalx=3;
        goaly=2;
        init_x_pos=0;
        init_y_pos=0;

    }
    
    
    if(environment==2)
    {
    
        height_grid= 4;
        width_grid=12;
        goalx=11;
        goaly=0;
        init_x_pos=0;
        init_y_pos=0;

    }
    
    
    for(i=0; i < width_grid; i++)
    {
        for(j=0; j< height_grid; j++)
        {
            if(environment==1)
            {
                reward[i][j]=-0.04;
                blocked[i][j]=0;
                
            }
            
            if(environment==2)
            {
                reward[i][j]=-1;
                blocked[i][j]=0;
            }
            
            for(k=0; k<4; k++)
            {
                Qvalues[i][j][k]=rand()%10;
                cout << "Initial Q value of cell [" <<i << ", " <<j << "] action " << k << " = " << Qvalues[i][j][k] << "\n";
            }
            
        }
        
    }
    
    if(environment==1)
    {
        reward[goalx][goaly]=1.0;
        reward[goalx][(goaly-1)]=-1.0;
        blocked[1][1]=1;
    }
    
    if(environment==2)
    {
        reward[goalx][goaly]=1;
        
        for(int h=1; h<goalx;h++)
        {   
            reward[h][0]=-100;
            
        }
        
    }
    
}
//PRINTEAR MATRIZ
void print_reward_grid() {
    printf("Matriz de recompensas:\n");
    for(int y = height_grid - 1; y >= 0; y--) {
        for(int x = 0; x < width_grid; x++) {
            if(blocked[x][y] == 1) {
                printf("  XX  ");
            } else {
                printf("%5.2f ", reward[x][y]);
            }
        }
        printf("\n");
    }
}


int action_selection()
{ // Based on the action selection method chosen, it selects an action to execute next

    
    if(action_sel==1) //Greedy, always selects the action with the largest Q value
    {
        
        int best_action = 0;
        float best_value = Qvalues[x_pos][y_pos][0];
        for(int a=1; a<4; a++) {
            if(Qvalues[x_pos][y_pos][a] > best_value) {
                best_value = Qvalues[x_pos][y_pos][a];
                best_action = a;
            }
        }
        return best_action;
    }
    
    if(action_sel==2)//epsilon-greedy, selects the action with the largest Q value with prob (1-exp_rate) and a random action with prob (exp_rate)
    {
        float r = ((float) rand() / (RAND_MAX));
        
        if(r < exp_rate) {
            return rand()%4; // Exploración
        } else {
            int best_action = 0;
            float best_value = Qvalues[x_pos][y_pos][0];
            for(int a=1; a<4; a++) {
                if(Qvalues[x_pos][y_pos][a] > best_value) {
                    best_value = Qvalues[x_pos][y_pos][a];
                    best_action = a;
                }
            }
            return best_action;
        }
        
    }
    return 0;
}

void move(int action)
{
    prev_x_pos=x_pos; //Backup of the current position, which will become past position after this method
    prev_y_pos=y_pos;
    
    //Stochastic transition model (not known by the agent)
    //Assuming a .8 prob that the action will perform as intended, 0.1 prob. of moving instead to the right, 0.1 prob of moving instead to the left
    
    if(stochastic_actions)
    {
        //Code here should change the value of variable action, based on the stochasticity of the action outcome
    }
    
    //After determining the real outcome of the chosen action, move the agent
    
    if(action==0) // Up
    {
        
        if((y_pos<(height_grid-1))&&(blocked[x_pos][y_pos+1]==0)) //If there is no wall or obstacle Up from the agent
        {
            y_pos=y_pos+1;  //move up
        }
        
    }
    
    
    if(action==1)  //Right
    {
        
        if((x_pos<(width_grid-1))&&(blocked[x_pos+1][y_pos]==0)) //If there is no wall or obstacle Right from the agent
        {
            x_pos=x_pos+1; //Move right
        }
        
    }
    
    if(action==2)  //Down
    {
        
        if((y_pos>0)&&(blocked[x_pos][y_pos-1]==0)) //If there is no wall or obstacle Down from the agent
        {
            y_pos=y_pos-1; // Move Down
        }
        
    }
    
    if(action==3)  //Left
    {
        
        if((x_pos>0)&&(blocked[x_pos-1][y_pos]==0)) //If there is no wall or obstacle Left from the agent
        {
            x_pos=x_pos-1;//Move Left
        }
        
    }
  }

void update_q_prev_state() //Updates the Q value of the previous state
{
    
    //Determine the max_a(Qvalue[x_pos][y_pos])
    
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))) )
    {
        //Calcula el maximo del sigueinte estado
        float maxQ = Qvalues[x_pos][y_pos][0];
        //Acciones posibles
        for (int actions = 1; actions < 4; actions++)
        {
            if (Qvalues[x_pos][y_pos][actions] > maxQ){
                maxQ = Qvalues[x_pos][y_pos][actions];
            }
        }

        //ECUACION Q VALUES Q(s, a) = Q(s, a) + α [r + γ * max_a' Q(s', a') - Q(s, a)]

        Qvalues[prev_x_pos][prev_y_pos][action_taken]= Qvalues[prev_x_pos][prev_y_pos][action_taken]
        + learn_rate *(reward[x_pos][y_pos] + disc_factor * maxQ - Qvalues[prev_x_pos][prev_y_pos][action_taken]);

    }
    else//Update the Q value of the previous state and action if the agent has reached a terminal state
    {
        //SI ES TERMINAL
        Qvalues[prev_x_pos][prev_y_pos][action_taken]=  Qvalues[prev_x_pos][prev_y_pos][action_taken]
        + learn_rate * (reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken]);

    }
}

void update_q_prev_state_sarsa()
{
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
    {
        
       Qvalues[prev_x_pos][prev_y_pos][action_taken]= Qvalues[prev_x_pos][prev_y_pos][action_taken];
    }
    else//Update the Q value of the previous state and action if the agent has reached a terminal state
    {
       Qvalues[prev_x_pos][prev_y_pos][action_taken]= Qvalues[prev_x_pos][prev_y_pos][action_taken];
    }
    
    
}

void Qlearning()
{
   // Seleccion de accion a ejecutar y se mueve
   move(action_selection());

   //Recompensa
   cum_reward=cum_reward+reward[x_pos][y_pos]; //Add the reward obtained by the agent to the cummulative reward of the agent in the current episode
    update_q_prev_state();    
}

void Sarsa()
{
    move(action_selection());
    cum_reward=cum_reward+reward[x_pos][y_pos]; //Add the reward obtained by the agent to the cummulative reward of the agent in the current episode

    //Actualizar Q value
    update_q_prev_state();
    
    
}

void Multi_print_grid()
{
    int x, y;
    
    for(y = (height_grid-1); y >=0 ; --y)
    {
        for (x = 0; x < width_grid; ++x)
        {

            if(blocked[x][y]==1) {
                cout << " \033[42m# \033[0m";
               
            }else{
                if ((x_pos==x)&&(y_pos==y)){
                    cout << " \033[44m1 \033[0m";
                    
                }else{
                    cout << " \033[31m0 \033[0m";
                        
                    
                }
            }
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    Initialize_environment();//Initialize the features of the chosen environment (goal and initial position, obstacles, rewards)
    print_reward_grid();
    for(i=0;i<num_episodes;i++)
    {
        //cout << "\n \n Episode " << i;
        current_episode=i;
        x_pos=init_x_pos;
        y_pos=init_y_pos;
        cum_reward=0;

        //If Sarsa was chosen as the algorithm:
        if(algorithm==2)
        {
         action_taken= action_selection();
            
        }
        
        //While the agent has not reached a terminal state:
        while(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
        {
            if(algorithm==1)
            {
            Qlearning();
            }
            if(algorithm==2)
            {
            Sarsa();
            }
            
        }

        finalrw[i]=cum_reward;
        //cout << " Total reward obtained: " <<finalrw[i] <<"\n";      
    }
    // Generar el CSV con episodios y recompensas
    std::ofstream csv_output("rewards_per_episode.csv");
    csv_output << "episode,reward_obtained\n";
    for(int ep=0; ep<num_episodes; ep++) {
        csv_output << ep << "," << finalrw[ep] << "\n";
    }
    csv_output.close();

    return 0;
}

