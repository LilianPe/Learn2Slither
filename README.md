# 🐍 Learn2Slither — Reinforcement Learning (42 Project)

## 🧠 Overview
**Learn2Slither** is a reinforcement learning project where an agent learns to play **Snake** autonomously using **Q-learning**.  
The goal is to train an AI capable of surviving and growing by making optimal decisions based only on **limited vision**.

---

## ✨ Features

- 🧠 **Reinforcement Learning Agent** — Learns to play Snake using a **Neural Q-network (Deep Q-Learning)**.  
- 🎯 **Reward System** — Positive and negative rewards for each action (eat apple, crash, idle, etc.).  
- 🧩 **ε-greedy Policy** — Balances exploration (random moves) and exploitation (best-known moves).  
- 👁️ **Limited Vision** — The snake only sees in 4 directions (UP, DOWN, LEFT, RIGHT).  
- 💾 **Model Checkpointing** — Save and load learning states to continue training later.  
- 🔄 **Multi-session Training** — Train over multiple sessions (1, 10, 100...) to track improvement.  
- 🚫 **Non-learning Mode** — Run the trained agent without updating Q-values (for evaluation).  
- ⚡ **Headless Training** — Option to disable graphical output to speed up training.  

## 🕹️ How to Launch

### ⚙️ Initialize the Project

Clone the repository and navigate into it, then run:
```bash
make
.venv/bin/activate```

### 🧮 Training Mode
Run several sessions and save your trained model:
```bash
./snake -sessions 10 -save models/10sess.txt -visual off```

### 🧠 Evaluation Mode (No Learning)

Load a trained model and observe the agent’s behavior:
```bash
./snake -visual on -load models/100sess.txt -dontlearn -step-by-step```
