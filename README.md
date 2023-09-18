# Deep-Reinforcement-Learning-in-Action-with-PyTorch
PyTorch implementations of deep reinforcement learning algorithms and environments


## 安装说明

支持Gym==0.25.2

创建Conda环境
```bash
conda create -n universe python=3.8
conda activate universe
pip install -r requirements.txt
```

## 可视化

```bash
tensorboard --logdir=/Users/cuiliu/Desktop/workspace/Deep-Reinforcement-Learning-in-Action-with-PyTorch/cross_entropy/runs --port=8080
```

# 开源算法库
- [rllab](https://github.com/rll/rllab)
- [Baseline](https://github.com/openai/baselines)
- [Stable Baselines]( https://github.com/hill-a/stable-baselines)
- [keras-rl](https://github.com/keras-rl/keras-rl)
- [BURLAP](http://burlap.cs.brown.edu/)
- [PyBrain](http://pybrain.org/)
- [RLPy](http://acl.mit.edu/RLPy/)
- [A Matlab Toolbox for Approximate RL and DP](http://busoniu.net/files/repository/readme_approxrl.html)

# 实战项目
### Implementation of Algorithms
- [Pytorch Implementation of DQN / DDQN / Prioritized replay/ noisy networks/ distributional values/ Rainbow/ hierarchical RL](https://github.com/higgsfield/RL-Adventure)
- [PyTorch implementations of various DRL algorithms for both single agent and multi-agent](https://github.com/ChenglongChen/pytorch-madrl)
- [Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl)
- [PyTorch 实现 DQN, AC, A2C, A3C, , Policy Gradient, DDPG, TRPO, PPO, ACER](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch)
- [Deep Reinforcement learning framework](https://github.com/VinF/deer)
- [Codes for understanding Reinforcement Learning( updating... )](https://github.com/halleanwoo/ReinforcementLearningCode)
- [Contains high quality implementations of Deep Reinforcement Learning algorithms written in PyTorch ](https://github.com/qfettes/DeepRL-Tutorials)
- [Implementation of Reinforcement Learning Algorithms. Python, OpenAI Gym, Tensorflow. Exercises and Solutions to accompany Sutton's Book and David Silver's course](https://github.com/dennybritz/reinforcement-learning)
- [Repo for the Deep Reinforcement Learning Nanodegree program](https://github.com/udacity/deep-reinforcement-learning)
- [教程 | 如何在Unity环境中用强化学习训练Donkey Car](https://mp.weixin.qq.com/s/DryUnnWXRnuAgyF6FvCjIg)
- [深入浅出解读"多巴胺（Dopamine）论文"、环境配置和实例分析](https://mp.weixin.qq.com/s/1iMjDZwdLLxsoUUqxk1XCQ)

# 论文
- [DQN-arxiv](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (Deep Q-Networks ): Mnih et al, 2013
    - [DQN-nature](https://www.nature.com/articles/nature14236)(Deep Q-Network ); Mnih et al, 2015 
    - [Double DQN](https://arxiv.org/abs/1509.06461) (Double Q Network) : Hasselt et al, 2015
    - [Dueling DQN](https://arxiv.org/abs/1511.06581) (Duling Q Network) : Ziyu Wang et al, 2015 
    - [QR-DQN](https://arxiv.org/abs/1710.10044) (Quantile Regression DQN): Dabney et al, 2017
- [Alpha Go](http://www.nature.com/nature/journal/v529/n7587/abs/nature16961.html)(Mastering the game of Go with deep neural networks and tree search) 
    - [AlphaZero-arxiv](https://arxiv.org/abs/1712.01815) (Mastering Chess and Shogi by Self-Play) :Silver et al, 2017 
    - [AlphaZero-nature](https://www.nature.com/articles/nature24270) (Go without human knowledge) :Silver et al, 2017
- [SAC](https://arxiv.org/abs/1801.01290) (Off-Policy Maximum Entropy): Haarnoja et al, 2018
    - [SAC](https://arxiv.org/abs/1812.05905) (Algorithms and Applications) : Haarnoja, et al 2018
- [A2C / A3C](https://arxiv.org/abs/1602.01783) (Asynchronous Advantage Actor-Critic): Mnih et al, 2016 
- [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization): Schulman et al, 2017
- [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization): Schulman et al, 2015
- [DPG](http://proceedings.mlr.press/v32/silver14.pdf) (Deterministic Policy Gradient) : DavidSilver et al, 2014
- [DDPG](https://arxiv.org/abs/1509.02971) (Deep Deterministic Policy Gradient): Lillicrap et al, 2015
- [TD3](https://arxiv.org/abs/1802.09477) (Twin Delayed DDPG): Fujimoto et al, 2018
- [NAF](https://arxiv.org/pdf/1603.00748v1.pdf) (Normalized adantage functions) : ShixiangGu et al, 2016
- [C51](https://arxiv.org/abs/1707.06887) (Categorical 51-Atom DQN): Bellemare et al, 2017
- [HER](https://arxiv.org/abs/1707.01495) (Hindsight Experience Replay): Andrychowicz et al, 2017
- [World Models](https://worldmodels.github.io/) Ha and Schmidhuber, 2018
- [I2A](https://arxiv.org/abs/1707.06203) (Imagination-Augmented Agents): Weber et al, 2017
- [MBMF](https://sites.google.com/view/mbmf) (Model-Based RL with Model-Free Fine-Tuning): Nagabandi et al, 2017
- [MBVE](https://arxiv.org/abs/1803.00101) (Model-Based Value Expansion): Feinberg et al, 2018
- [PathNet](https://arxiv.org/pdf/1701.08734.pdf)(Evolution Channels Gradient Descent):  Fernando et al, 2017
- [plannet](https://github.com/google-research/planet)(Learning Latent Dynamics) : Hafner, et al, 2018
- [TCN](https://arxiv.org/abs/1704.06888v1) (Time-Contrastive Networks):Sermanet, et al, 2017
- [Reinforcement and Imitation Learning](https://arxiv.org/pdf/1802.09564.pdf) : Yuke Zhu†, et al 2018
- [Prioritized experience replay](https://arxiv.org/abs/1511.05952):Schaul, et al 2015
- [Policy distillation](https://arxiv.org/abs/1511.06295) : Rusu, et al 2015
- [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868v2.pdf) : Bellemare, et al 2015
- [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](https://arxiv.org/pdf/1507.00814v3.pdf) : Stadie, et al 2015 
- [Action-Conditional Video Prediction using Deep Networks in Atari Games]( https://arxiv.org/pdf/1507.08750v2.pdf) : JunhyukOh, et al 2015
- [Control of Memory, Active Perception, and Action in Minecraft]( https://web.eecs.umich.edu/~baveja/Papers/ICML2016.pdf) : JunhyukOh, et al 2015

