# Intro

This project looks at implementing the [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971.pdf) (DDPG) algorithm in order to solve the Reacher environment.

This implementation used the ddpg pendulum example from the Udacity deep reinforcment learning repository as a foundation and was adapted to suite the Reacher environment. This project can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

The Reacher environment provided an interesting challenge and required several subtle but integral changes to the my initial code in order for the agent to train successfully.

# The Environment

![Trained Agent](/images/trained_maddpg.gif)

The environment used for this project was built using the Unity [ml-agents](https://github.com/Unity-Technologies/ml-agents) framework.

In this environment, two independent agents each control a paddle and are tasked with rallying the ball back and forth for as long as possible. A reward of +0.1 is given each time the agent hits the ball over the net. A reward of -0.1 is given to the agent each time the ball hits their side of the court.

The observation space consists of 8 variables corresponding to position and velocity of the ball and agents paddle. 

The action space consists of 2 continuous actions corresponding to movement toward net or away from net, and jumping.

# Algorithm Used - MADDPG/COMA

Initially I chose to go with Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm as the OpenAI paper [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)[5] shows that MADDPG outperforms many other RL algorithms. Here the paper looks at DQN , TRPO and DDPG. The results shows that MADDPG is able to not only converge faster, but is also more stable during training. It is worth pointing out that current state of the art algorithms such as PPO and SAC were not tested at the time of writing. Although these algorithms are capable of perform as well as, or if not better, than MADDPG, due to my past experiences with standard DDPG for similar control tasks I felt that MADDPG was a good algorithm to apply. However a key feature of MADDPG is the centralized critic networks for each agent. This used to allow multi agents to learn from different rewards, thus allowing MADDPG to be used in environments that can only use local information, does not assume a differentiable model of the environment dynamics and most importantly, allows the agent to be trained in bot competitive and coopoerative environments. This is very powerful, however as the Tennis environment uses the same reward function for all agents this level of generalisation is unnecessary. As such I felt that the approach used in the paper [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/pdf/1705.08926.pdf)[4] is better suited to this problem. In this paper researchers at Oxford took a different approach to the multi agent problem and used a shared critic between the agents. This provides a counterfactual baseline that improves both the speed of training and performance of multi agents. In my experiments I implemented an MADDPG using a single shared critic in order to improve training based on this environment. 

This report will go through the methodology of my implementation, the experiments and changes I made during training, the results achieved from these experiments and finally my thoughts on future work.

## Methodology

DDPG is an off-policy, actor critic method that performs more like an advanced DQN built for continuous tasks. Although DQN has achieved superhuman performance on a number of environments such as the Atari games, it can only handle discrete and low-dimensional action spaces[1]. As such it cant handle continuous action spaces. This problem can be solved by combining the techniques used in DQN with the actor-critic methodology. This new form of algorithm is capable of tackeling a much broader range of more complicated tasks such as the control problem provided in the Tennis environment. The next key component is the modification to allow for multi agents to learn to interact in the same environment. 

### DDPG 
![DDPG Pseudo Code](/images/DDPG_Psuedo.png)

As stated previously it is impractical to try and map Q values to state/actions for continuous tasks as Q learning requires an optimization step at each time step. This step is simply too slow for large, unconstrained function approximators and nontrivial action spaces[1]. Instead DDPG uses the actor network to maintain the current deterministic policy using the actor function μ(s|θμ) which maps states to the best action. Just like in Q learning, the critic is learned using the Q function Q(s|a) to determine the state/action value. During this calculation the critic takes in the output of the actor as target for training, similar to the approach used in DQN.

![Actor Critic Exampl](https://camo.githubusercontent.com/93fecaeda4aa38d024fa35b8d5e1b13329a9ea21/68747470733a2f2f7777772e73746576656e737069656c626572672e6d652f70726f6a656374732f696d616765732f646470675f747261696e2e676966)

As stated previously, DDPG builds upon much of the methodology used in DQN. As this is an off-policy method, DDPG uses experience replay to store large amounts of timesteps (s,a,r,s') and samples minibatches of these experiences in order to leanr/update the agents networks.

One problem seen in many experiments using Q learning with neural networks is that the training can be unstable. This is due to the fact that the network being updated Q(s,a|θQ) is also used in calculating the target value [1]. DQN (Mnih et al., 
) solved this using a target network, this was further improved on by using Double DQN's[3]. DDPG has a different approach to this problem using "soft" target updates instead of directly copying weights [1]. As in DQN, two networks are maintained for both the actor and critic, the local and target network. Instead of directly copying weights at regular intervals, soft updates mixes in 0.01% of the local network into the target network. This makes the network change slowly and greatly improves training stability.

The final problem associated with continuous action spaced environments is that of exploration. DQN uses an epsilon-greedy approach, that works well for discrete action spaces but is not sufficient for continuous action spaces. Instead DDPG  constructs an exploration policy μ′ by adding noise sampled from a noise process N to our actor policy[1]. The noise used here is the Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930).

      μ′(st) = μ(st|θtμ) + N

### MADDPG Psuedo
<img src="/images/MADDPG_Psuedo.jpg" alt="MADDPG Pseudo Code" width="500"/>

### MADDPG Multi Critics
<img src="/images/multi_critic.png" alt="Multi Critics" width="500" />

### COMA Single Critic
<img src="/images/single_critic.png" alt="Single Critics" width="500"/>
      


## Experiments and Training

One of the main appeals of the DDPG algorithms is the simplicity of its implementation while still providing state of the art results. Although the implementation is simple, fine tuning the architecture and parameters of the agent can still be a difficult and time consuming task. During my implementation of the algorithm I ran into several problems that were eventaully solved by making changes to several key areas.

**Updates**

As seen in the pseudo code adove, the original paper shows that the algorithm performs an update step at each time step. When trying this in my initial experiments the agent was unable to get passed an average score of 1. During my research that other people tried to periodically update the agent, for example every 10 timesteps the agent would 20 update steps. This was in an attempt to update with a larger and more diverse sample. I experimented with several variations of update frequency and iterations. Eventually I tried only updating the agent at the end of an episode for 10 timesteps and this proved to provide the most stable results. I am still unsure why this performed better than simply updating at each timestep as seen in the paper. My theory is that by waiting for the end of the episode there is a more diverse experience buffer to sample from which  improves the stability of the agent. 

**Catastrophic Forgetting**

A common problem when training neural networks is that convergence and stabilty is not garunteed. This is painfully displayed in the phenomenon known as "catastrophic forgetting". This is when a network continues to improve until it will begin to (sometimes suddenly) deteriorate. This is shown in the graph below.

![Catastrophic Forgetting](images/catastrophic_forgetting.png)

It is still unclear what exactly causes this convergence on a poor policy but is likely a combination of poor hyper-parameters. During my research and experiments I discovered that the samples used during the update step was critical for fixing this issue. By increase the size of my experience replay buffer to 10000000 and my batch size to 512 the agents training stablized. I believe this comes back to the problem seen in my update steps of diverse samples. By increasing the amount of experiences store and the sample size, the agent is able to update and learn on more generalised data. This reduces the possibility of the agent getting stuck in a local minima due to poor experience replay.

**Model Architecture**

The final piece of the puzzle was finding the correct network architecture for the reacher agent. I experimented with both the initial architecture found in the Udacity example which was an MLP with 2 layers of 400 and 300 nodes each with a ReLU layer. Both the Actor and Critic used the same network architecture This model showed promise and was able to learn, but was incredibly slow, taking more than 50 episodes to get past an average score of 1. The next experiment was to use the same architecture found in DeepMinds paper[link] which was the same for the critic but used 300 and 200 nodes respectively for the actore layers. Again, this showed that the agent was learning, but was very slow. This could be that the initial training of the agent was slow but after an initial period would pick up. 

I began to suspect that the agent was unable to accurately assign credit correctly due to the (relative) complexity of the environment. In order to solve this I began experimenting with much larger networks. Eventually I settled on the follow network architecture and provided the best results for this environment.

| Layers           |Parameters           |
|:-------------:| :-------------:| 
| Dense Layer| 1000| 
| ReLU activation| NA|   
| Dense Layer| 1000| 
| ReLU activation| NA| 
|Dense Layer|4| 

# Results

I conclusion, DDPG was successfully able to converge on an optimal policy that was capable of reaching and maintaining an average score of 39.4 . This required some tweaking of the update parameters and large network than described in literature but achieved good results. The full results and hyperparameters used are shown below. As you can see from the graph, the agent is slow to learn at first, but after ~50 episodes the learn improves drastically. As you can see from the plot below, the agent began hitting a score 39/40 after ~100 episodes and reached the competion score of 30 after ~70 episodes. Of course it took longer to get the 100 episode average to prove that the agents score was stable.

Based on this I believe that other architectures, such as the one described in the DDPG paper, may work equally well but I simply did let the agent train for long enough. Nonetheless, I am very happy with the results of my agent.

![DDDQN](/images/ddpg_40.png)

| Parameter | Value |  
|:-------------:| :-------------:|
|Layer 1    | 1000     | 
|Layer 2    | 1000     |  
|Learning Rate Actor    | 0.0001   |  
|Learning Rate Critic   | 0.0001   |  
|Weight Decay | 0 |
|Batch Size    | 512     | 
|Buffer Size    | 1000000    | 
|Update Size | 10 |
| Tau | 0.001|
|Gamma| 0.99|


# Future Work

## D4PG
The obvious next step would be to try to implement the successor of DDPG, Distributed Distributional Deterministic Policy Gradient. This is a much more advanced algorithm and has been shown to significantly improve upon DDPG for most tasks under most conditions. 

## Crawler Environment
With the success of the agent in the reacher environment I attempted to train it on the more advanced crawler environnment. Unfortunately the crawler binary was not able to run when I began training. I am not sure why this is the case and I am waiting for a response from the Udacity team on the issue. Once I get a working copy of the binary I will begin training my DDPG agent on the crawler.

# References


[1] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In Inter- national Conference on Learning Representations, 2016.

[2] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez,
Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, Timothy Lillicrap, Martin Riedmiller. DeepMind Control Suite. cs.AI, 2018

[3] Hado van Hasselt, Arthur Guez, David Silver. Deep Reinforcement Learning with Double Q-learning. 2016

[4] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson. Counterfactual multi-agent
policy gradients. arXiv preprint arXiv:1705.08926, 2017.

[5] R. Lowe, Y. Wu, A. Tamar, J. Harb, P, AAbbeel, I, Mordatch .Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. arXiv preprint arXiv:1706.02275, 2017
