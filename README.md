# Awesome Agent

<!-- badge link https://github.com/badges/awesome-badges -->
<!-- [![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re) -->
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![issues](https://custom-icon-badges.herokuapp.com/github/issues-raw/weleen/awesome-agent?logo=issue)](https://github.com/weleen/awesome-agent/issues "issues")
[![license](https://custom-icon-badges.herokuapp.com/github/license/weleen/awesome-agent?logo=law&logoColor=white)](https://github.com/weleen/awesome-agent/blob/main/LICENSE?rgh-link-date=2021-08-09T18%3A10%3A26Z "license MIT")
[![stars](https://custom-icon-badges.herokuapp.com/github/stars/weleen/awesome-agent?logo=star)](https://github.com/weleen/awesome-agent/stargazers "stars")

This repo collects papers and repositories about agent. Especially, we focus on the agent that can be applied to the real-world tasks, such as robotics, search engine, scientific discovery, etc. Please feel free to PR the works missed by the repo. We use the following format to record the papers:
```md
[Paper Title](paper link), Conference/Journal, Year, Team | [code](code link), [project](project link)
```

## TODO

- [ ] Update CoRL2024 papers.
- [ ] Update IROS2024 papers.
- [ ] Update ICRA2024 papers.

## Table of Contents

- [Awesome Agent](#awesome-agent)
  - [TODO](#todo)
  - [Table of Contents](#table-of-contents)
    - [Foundation Model (LLM, VLM, etc.)](#foundation-model-llm-vlm-etc)
    - [Robotics](#robotics)
    - [Scientific Discovery](#scientific-discovery)
    - [Multi-Agent](#multi-agent)
    - [Others](#others)
  - [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [General AI Agent](#general-ai-agent)
    - [Robotics](#robotics-1)
  - [Other Resources](#other-resources)
    - [Github Collections](#github-collections)
    - [Tutorial](#tutorial)
    - [Group](#group)
    - [Frameworks](#frameworks)
      - [Foundation Models](#foundation-models)
      - [Multi-Agent](#multi-agent-1)
    - [Links](#links)
    - [Star History](#star-history)

### Foundation Model (LLM, VLM, etc.)

<details><summary>2024</summary>

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf), Preprint, 2025.1, DeepSeek | [code](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf), Preprint, 2024.12, DeepSeek | [code](https://github.com/deepseek-ai/DeepSeek-V3)
- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/pdf/2408.03326), ArXiv, 2024.8, NTU & Bytedance | [project](https://llava-vl.github.io/blog/2024-08-05-llava-onevision), [code](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation](https://arxiv.org/pdf/2409.04429), ArXiv, 2024.9, THU & MIT & NVIDIA & UCB & UCSD. | [project](https://hanlab.mit.edu/projects/vila-u), [code](https://github.com/mit-han-lab/vila-u)
- [Qwen2 & QwQ: Reflect Deeply on the Boundaries of the Unknown](https://arxiv.org/abs/2407.10671), ArXiv, 2024.7 & 2024.11, Qwen Team at Alibaba | [model](https://huggingface.co/Qwen/QwQ-32B-Preview), [project](https://huggingface.co/Qwen), [blog](https://qwenlm.github.io/blog/qwq-32b-preview/)
- [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/abs/2411.14405), ArXiv, 2024.11, AIDC-AI at Alibaba | [model](https://huggingface.co/AIDC-AI/Marco-o1), [code](https://github.com/AIDC-AI/Marco-o1)
- [O1 Journey: O1 Replication Journey – Part 2: Surpassing O1-preview through Simple Distillation Big Progress or Bitter Lesson?](https://arxiv.org/abs/2411.16489), ArXiv, 2024.11, GAIR SJTU | [code](https://github.com/GAIR-NLP/O1-Journey)
- [Gemma: Open models based on gemini research and technology](https://arxiv.org/abs/2403.08295), ArXiv, 2024.3, Google | [project](https://blog.google/technology/developers/gemma-open-models/)
- [Minicpm: Unveiling the potential of small language models with scalable training strategies](https://arxiv.org/pdf/2404.06395), CoLM, 2024, OpenBMB. | [blog](https://openbmb.vercel.app/?category=Blog), [code](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/pdf/2408.01800), ArXiv, 2024, OpenBMB. | [code](https://github.com/OpenBMB/MiniCPM-V)
 </details>
 

<details><summary>2023</summary>

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/pdf?id=WE_vluYUL-X), ICLR, 2023, Princeton & Google | [project](https://react-lm.github.io/), [code](https://github.com/ysymyth/ReAct)
</details>

### Robotics

<details><summary>2024</summary>

- [AgiBot World](https://agibot-world.com/), Zhiyuan Robot | [dataset](https://huggingface.co/agibot-world), [code](https://github.com/OpenDriveLab/AgiBot-World)
- [Genesis: A Generative and Universal Physics Engine for Robotics and Beyond](https://genesis-embodied-ai.github.io/), ArXiv, 2024. | [code](https://github.com/Genesis-Embodied-AI/Genesis), [project](https://genesis-embodied-ai.github.io/)
- [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning](https://arxiv.org/pdf/2302.02662), ICML, 2024, Inria. | [code](https://*github*.com/flowersteam/Grounding_LLMs_with_online_RL)
- [True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning](https://arxiv.org/abs/2401.14151), ICLR2024, NTU. | [code](https://github.com/WeihaoTan/TWOSOME)
- [Stein Variational Ergodic Search](/2024/program/papers/1/), RSS, 2024
- [Parallel and Proximal Linear-Quadratic Methods for Real-Time Constrained Model-Predictive Control](/2024/program/papers/2/), RSS, 2024
- [Differentiable Robust Model Predictive Control](/2024/program/papers/3/), RSS, 2024
- [Computation-Aware Learning for Stable Control with Gaussian Process](/2024/program/papers/4/), RSS, 2024
- [Decentralized Multi-Robot Line-of-Sight Connectivity Maintenance under Uncertainty](/2024/program/papers/5/), RSS, 2024
- [Hamilton-Jacobi Reachability Analysis for Hybrid Systems with Controlled and Forced Transitions](/2024/program/papers/6/), RSS, 2024
- [JIGGLE: An Active Sensing Framework for Boundary Parameters Estimation in Deformable Surgical Environments](/2024/program/papers/7/), RSS, 2024
- [Conformalized Teleoperation: Confidently Mapping Human Inputs to High-Dimensional Robot Actions](/2024/program/papers/8/), RSS, 2024
- [Optimal Non-Redundant Manipulator Surface Coverage with Rank-Deficient Manipulability Constraints](/2024/program/papers/9/), RSS, 2024
- [AdaptiGraph: Material-Adaptive Graph-Based Neural Dynamics for Robotic Manipulation](/2024/program/papers/10/), RSS, 2024
- [Human-oriented Representation Learning for Robotic Manipulation](/2024/program/papers/11/), RSS, 2024
- [Dynamic On-Palm Manipulation via Controlled Sliding](/2024/program/papers/12/), RSS, 2024
- [Efficient Data Collection for Robotic Manipulation via Compositional Generalization](/2024/program/papers/13/), RSS, 2024
- [Demonstrating Learning from Humans on Open-Source Dexterous Robot Hands](/2024/program/papers/14/), RSS, 2024
- [Reconciling Reality through Simulation: A Real-To-Sim-to-Real Approach for Robust Manipulation](/2024/program/papers/15/), RSS, 2024
- [SAGE: Bridging Semantic and Actionable Parts for GEneralizable Articulated-Object Manipulation under Language Instructions](/2024/program/papers/16/), RSS, 2024
- [Demonstrating Event-Triggered Investigation and Sample Collection for Human Scientists using Field Robots and Large Foundation Models](/2024/program/papers/17/), RSS, 2024
- [CraterGrader: Autonomous Robotic Terrain Manipulation for Lunar Site Preparation and Earthmoving](/2024/program/papers/18/), RSS, 2024
- [POAM: Probabilistic Online Attentive Mapping for Efficient Robotic Information Gathering](/2024/program/papers/19/), RSS, 2024
- [Blending Data-Driven Priors in Dynamic Games](/2024/program/papers/20/), RSS, 2024
- [Demonstrating HOUND: A Low-cost Research Platform for High-speed Off-road Underactuated Nonholonomic Driving](/2024/program/papers/21/), RSS, 2024
- [Model Predictive Control for Aggressive Driving Over Uneven Terrain](/2024/program/papers/22/), RSS, 2024
- [Demonstrating CropFollow++: Robust Under-Canopy Navigation with Keypoints](/2024/program/papers/23/), RSS, 2024
- [SEEK: Semantic Reasoning for Object Goal Navigation in Real World Inspection Tasks](/2024/program/papers/24/), RSS, 2024
- [Yell At Your Robot: Improving On-the-Fly from Language Corrections](/2024/program/papers/25/), RSS, 2024
- [Task Adaptation in Industrial Human-Robot Interaction: Leveraging Riemannian Motion Policies](/2024/program/papers/26/), RSS, 2024
- [Risk-Calibrated Human-Robot Interaction via Set-Valued Intent Prediction](/2024/program/papers/27/), RSS, 2024
- [Constraint-Aware Intent Estimation for Dynamic Human-Robot Object Co-Manipulation](/2024/program/papers/28/), RSS, 2024
- [Demonstrating HumanTHOR: A Simulation Platform and Benchmark for Human-Robot Collaboration in a Shared Workspace](/2024/program/papers/29/), RSS, 2024
- [Developing Design Guidelines for Older Adults with Robot Learning from Demonstration](/2024/program/papers/30/), RSS, 2024
- [FLAIR: Feeding via Long-Horizon AcquIsition of Realistic dishes](/2024/program/papers/31/), RSS, 2024
- [The Benefits of Sound Resound: An In-Person Replication of the Ability of Character-Like Robot Sound to Improve Perceived Social Warmth](/2024/program/papers/32/), RSS, 2024
- [Leveraging Large Language Model for Heterogeneous Ad Hoc Teamwork Collaboration](/2024/program/papers/33/), RSS, 2024
- [INTERPRET: Interactive Predicate Learning from Language Feedback for Generalizable Task Planning](/2024/program/papers/34/), RSS, 2024
- [Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres](/2024/program/papers/35/), RSS, 2024
- [Motion Planning in Foliated Manifolds using Repetition Roadmap](/2024/program/papers/36/), RSS, 2024
- [Language-Augmented Symbolic Planner for Open-World Task Planning](/2024/program/papers/37/), RSS, 2024
- [Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Motion Planning with Pointclouds](/2024/program/papers/38/), RSS, 2024
- [Homotopic Path Set Planning for Robot Manipulation and Navigation](/2024/program/papers/39/), RSS, 2024
- [Practice Makes Perfect: Planning to Learning Skill Parameter Policies](/2024/program/papers/40/), RSS, 2024
- [World Models for General Surgical Grasping](/2024/program/papers/41/), RSS, 2024
- [SpringGrasp: Synthesizing Compliant, Dexterous Grasps under Shape Uncertainty](/2024/program/papers/42/), RSS, 2024
- [DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation](/2024/program/papers/43/), RSS, 2024
- [GRaCE: Balancing Multiple Criteria to Achieve Stable, Collision-Free, and Functional Grasps](/2024/program/papers/44/), RSS, 2024
- [Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots](/2024/program/papers/45/), RSS, 2024
- [Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering](/2024/program/papers/46/), RSS, 2024
- [Demonstrating Adaptive Mobile Manipulation in Retail Environments](/2024/program/papers/47/), RSS, 2024
- [Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning](/2024/program/papers/48/), RSS, 2024
- [RT-H: Action Hierarchies using Language](/2024/program/papers/49/), RSS, 2024
- [RoboCasa: Large-Scale Simulation of Household Tasks for Generalist Robots](/2024/program/papers/50/), RSS, 2024
- [Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning](/2024/program/papers/51/), RSS, 2024
- [Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers](/2024/program/papers/52/), RSS, 2024
- [Offline Imitation Learning Through Graph Search and Retrieval](/2024/program/papers/54/), RSS, 2024
- [RVT-2: Learning Precise Manipulation from Few Demonstrations](/2024/program/papers/55/), RSS, 2024
- [Imitation Bootstrapped Reinforcement Learning](/2024/program/papers/56/), RSS, 2024
- [Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers](/2024/program/papers/57/), RSS, 2024
- [Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning](/2024/program/papers/58/), RSS, 2024
- [Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion](/2024/program/papers/59/), RSS, 2024
- [RL2AC: Reinforcement Learning-based Rapid Online Adaptive Control for Legged Robot Robust Locomotion](/2024/program/papers/60/), RSS, 2024
- [HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation](/2024/program/papers/61/), RSS, 2024
- [MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting](/2024/program/papers/62/), RSS, 2024
- [Collaborative Planar Pushing of Polytopic Objects with Multiple Robots in Complex Scenes](/2024/program/papers/63/), RSS, 2024
- [AutoMate: Specialist and Generalist Assembly Policies over Diverse Geometries](/2024/program/papers/64/), RSS, 2024
- [An abstract theory of sensor eventification](/2024/program/papers/65/), RSS, 2024
- [Octopi: Object Property Reasoning with Large Tactile-Language Models](/2024/program/papers/66/), RSS, 2024
- [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](/2024/program/papers/67/), RSS, 2024
- [HRP: Human affordances for Robotic Pre-training](/2024/program/papers/68/), RSS, 2024
- [MIRAGE: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting](/2024/program/papers/69/), RSS, 2024
- [Broadcasting Support Relations Recursively from Local Dynamics for Object Retrieval in Clutters](/2024/program/papers/70/), RSS, 2024
- [Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation](/2024/program/papers/71/), RSS, 2024
- [CLOSURE: Fast Quantification of Pose Uncertainty Sets](/2024/program/papers/72/), RSS, 2024
- [GOAT: GO to Any Thing](/2024/program/papers/73/), RSS, 2024
- [Demonstrating Arena 3.0: Advancing Social Navigation in Collaborative and Highly Dynamic Environments](/2024/program/papers/74/), RSS, 2024
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Multi-Modal Large Language Model Learning](/2024/program/papers/75/), RSS, 2024
- [Dynamic Adversarial Attacks on Autonomous Driving Systems](/2024/program/papers/76/), RSS, 2024
- [Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation](/2024/program/papers/77/), RSS, 2024
- [ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection](/2024/program/papers/78/), RSS, 2024
- [NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation](/2024/program/papers/79/), RSS, 2024
- [RACER: Epistemic Risk-Sensitive RL Enables Fast Driving with Fewer Crashes](/2024/program/papers/80/), RSS, 2024
- [Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments](/2024/program/papers/81/), RSS, 2024
- [Demonstrating Agile Flight from Pixels without State Estimation](/2024/program/papers/82/), RSS, 2024
- [You’ve Got to Feel It To Believe It: Multi-Modal Bayesian Inference for Semantic and Property Prediction](/2024/program/papers/83/), RSS, 2024
- [AnyFeature-VSLAM: Automating the Usage of Any Feature into Visual SLAM](/2024/program/papers/84/), RSS, 2024
- [iMESA: Incremental Distributed Optimization for Collaborative Simultaneous Localization and Mapping](/2024/program/papers/85/), RSS, 2024
- [Scalable Distance-based Multi-Agent Relative State Estimation via Block Multiconvex Optimization](/2024/program/papers/86/), RSS, 2024
- [Experience-based multi-agent path finding with narrow corridors](/2024/program/papers/87/), RSS, 2024
- [Event-based Visual Inertial Velometer](/2024/program/papers/88/), RSS, 2024
- [Explore until Confident: Efficient Exploration for Embodied Question Answering](/2024/program/papers/89/), RSS, 2024
- [Octo: An Open-Source Generalist Robot Policy](/2024/program/papers/90/), RSS, 2024
- [Demonstrating OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics](/2024/program/papers/91/), RSS, 2024
- [Any-point Trajectory Modeling for Policy Learning](/2024/program/papers/92/), RSS, 2024
- [Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation](/2024/program/papers/93/), RSS, 2024
- [DrEureka: Language Model Guided Sim-To-Real Transfer](/2024/program/papers/94/), RSS, 2024
- [Set It Up!: Functional Object Arrangement with Compositional Generative Models](/2024/program/papers/95/), RSS, 2024
- [Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics](/2024/program/papers/96/), RSS, 2024
- [ConTac: Continuum-Emulated Soft Skinned Arm with Vision-based Shape Sensing and Contact-aware Manipulation](/2024/program/papers/97/), RSS, 2024
- [Function Based Sim-to-Real Learning for Shape Control of Deformable Free-form Surfaces](/2024/program/papers/98/), RSS, 2024
- [Safe & Accurate at Speed with Tendons: A Robot Arm for Exploring Dynamic Motion](/2024/program/papers/99/), RSS, 2024
- [Evolution and learning in differentiable robots](/2024/program/papers/100/), RSS, 2024
- [Construction of a Multiple-DOF Underactuated Gripper with Force-Sensing via Deep Learning](/2024/program/papers/101/), RSS, 2024
- [A Single Motor Nano Aerial Vehicle with Novel Peer-to-Peer Communication and Sensing Mechanism](/2024/program/papers/102/), RSS, 2024
- [Design and Control of a Bipedal Robotic Character](/2024/program/papers/103/), RSS, 2024
- [POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints](/2024/program/papers/104/), RSS, 2024
- [Demonstrating Language-Grounded Motion Controller](/2024/program/papers/105/), RSS, 2024
- [VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation](/2024/program/papers/106/), RSS, 2024
- [Expressive Whole-Body Control for Humanoid Robots](/2024/program/papers/107/), RSS, 2024
- [From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach](/2024/program/papers/108/), RSS, 2024
- [MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints](/2024/program/papers/109/), RSS, 2024
- [Linear-time Differential Inverse Kinematics: an Augmented Lagrangian Perspective](/2024/program/papers/110/), RSS, 2024
- [A Trajectory Tracking Algorithm for the LSMS Family of Cable-Driven Cranes](/2024/program/papers/111/), RSS, 2024
- [AutoGPT+P: Affordance-based Task Planning using Large Language Models](/2024/program/papers/112/), RSS, 2024
- [Implicit Graph Search for Planning on Graphs of Convex Sets](/2024/program/papers/113/), RSS, 2024
- [Real-Time Anomaly Detection and Reactive Planning with Large Language Models](/2024/program/papers/114/), RSS, 2024
- [iHERO: Interactive Human-oriented Exploration and Supervision Under Scarce Communication](/2024/program/papers/115/), RSS, 2024
- [Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots](/2024/program/papers/116/), RSS, 2024
- [Goal-Reaching Trajectory Design Near Danger with Piecewise Affine Reach-avoid Computation](/2024/program/papers/117/), RSS, 2024
- [Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness](/2024/program/papers/118/), RSS, 2024
- [Logic-Skill Programming: An Optimization-based Approach to Sequential Skill Planning](/2024/program/papers/119/), RSS, 2024
- [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](/2024/program/papers/120/), RSS, 2024
- [Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals](/2024/program/papers/121/), RSS, 2024
- [Don't Start From Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion](/2024/program/papers/122/), RSS, 2024
- [Learning Manipulation by Predicting Interaction](/2024/program/papers/123/), RSS, 2024
- [URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images](/2024/program/papers/124/), RSS, 2024
- [Learning to Learn Faster from Human Feedback with Language Model Predictive Control](/2024/program/papers/125/), RSS, 2024
- [Natural Language Can Help Bridge the Sim2Real Gap](/2024/program/papers/126/), RSS, 2024
- [PoCo: Policy Composition from and for Heterogeneous Robot Learning](/2024/program/papers/127/), RSS, 2024
- [Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand](/2024/program/papers/128/), RSS, 2024
- [HACMan++: Spatially-Grounded Motion Primitives for Manipulation](/2024/program/papers/129/), RSS, 2024
- [RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing](/2024/program/papers/130/), RSS, 2024
- [Configuration Space Distance Fields for Manipulation Planning](/2024/program/papers/131/), RSS, 2024
- [Towards Tight Convex Relaxations for Contact-Rich Manipulation](/2024/program/papers/132/), RSS, 2024
- [THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation](/2024/program/papers/133/), RSS, 2024
- [One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation](/2024/program/papers/134/), RSS, 2024
- [Tactile-Driven Non-Prehensile Object Manipulation via Extrinsic Contact Mode Control](/2024/program/papers/135/), RSS, 2024
- [Learning Multi-Agent Collaborative Manipulation for Long-Horizon Quadrupedal Pushing](https://arxiv.org/pdf/2411.07104), ArXiv, 2024, CMU & Google. | [project](https://collaborative-mapush.github.io/), [code](https://github.com/collaborative-mapush/MAPush)
- [ARMOR：Egocentric Perception for Humanoid Robot Collision Avoidance and Motion Planning](https://arxiv.org/abs/2412.00396), ArXiv, 2024.12. CMU & Apple
- [CASHER: Robot Learning with Super-Linear Scaling](https://arxiv.org/abs/2412.01770), ArXiv, 2024.12. MIT & UoW | [project](https://casher-robot-learning.github.io/CASHER/)
- [CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation](https://arxiv.org/abs/2411.19650), ArXiv, 2024.12, MSRA Asia | [project](https://cogact.github.io/), [code](https://github.com/microsoft/CogACT)
- [Vision-Language Foundation Models as Effective Robot Imitators](https://arxiv.org/abs/2311.01378), ICLR, 2024, Bytedance | [project](https://roboflamingo.github.io/), [code](https://github.com/RoboFlamingo/RoboFlamingo)
- [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](https://arxiv.org/pdf/2411.02359), ArXiv, 2024.11, Tsinghua & Bytedance | [code](https://github.com/yueyang130/DeeR-VLA)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf), ArXiv, 2024.10, Physical Intelligence | [project](https://physicalintelligence.company/blog/pi0), [code](https://github.com/PhysicalIntelligence/pi0)
- [GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation](https://arxiv.org/abs/2410.06158), ArXiv, 2024.10, Bytedance | [project](https://gr2-manipulation.github.io/)
- [Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers](https://arxiv.org/abs/2409.20537), ArXiv, 2024.9, MIT | [code](https://github.com/liruiw/HPT)
- [Mobile ALOHA Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation](https://mobile-aloha.github.io/resources/mobile-aloha.pdf), CoRL, 2024, Stanford | [project](https://mobile-aloha.github.io/), [code](https://github.com/MarkFzp/mobile-aloha), [code2](https://github.com/MarkFzp/act-plus-plus)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), ArXiv, 2024.6, Stanford | [project](https://openvla.github.io/), [code](https://github.com/openvla/openvla)
- [RoboUniView: Visual-Language Model with Unified View Representation for Robotic Manipulation](https://arxiv.org/pdf/2406.18977), ArXiv, 2024.6, Meituan | [project](https://liufanfanlff.github.io/RoboUniview.github.io/), [code](https://github.com/liufanfanlff/RoboUniview)
- [RoboDreamer: Learning Compositional World Models for Robot Imagination](https://arxiv.org/pdf/2404.12377), ArXiv, 2024.4, HKUST & MIT & UCSD & Google Research | [project](https://robovideo.github.io/)
- [RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches](https://arxiv.org/abs/2311.01977), ICLR, 2024, Google DeepMind | [project](https://rt-trajectory.github.io/)
- [ALOHA Unleashed 🌋: A Simple Recipe for Robot Dexterity](https://aloha-unleashed.github.io/assets/aloha_unleashed.pdf), ArXiv, 2024, Google DeepMind | [project](https://aloha-unleashed.github.io/)
- [HumanPlus Humanoid Shadowing and Imitation from Humans](https://humanoid-ai.github.io/HumanPlus.pdf), CoRL, 2024, Stanford | [project](https://humanoid-ai.github.io/), [code](https://github.com/MarkFzp/humanplus)
- [Universal Manipulation Interface In-The-Wild Robot Teaching Without In-The-Wild Robots](https://arxiv.org/abs/2402.10329), RSS, 2024, Stanford & Columbia | [project](https://umi-gripper.github.io/), [code](https://github.com/real-stanford/universal_manipulation_interface)
- [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](https://arxiv.org/abs/2403.03954), RSS, 2024, Shanghai Qizhi Institute & SJTU & Tsinghua & Shanghai AI Lab | [project](https://3d-diffusion-policy.github.io/), [code](https://github.com/columbia-ai-robotics/diffusion_policy)
- [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/pdf/2410.07864), ArXiv, 2024, Tsinghua | [project](https://rdt-robotics.github.io/rdt-robotics/)
- [Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals](https://www.roboticsproceedings.org/rss20/p121.pdf), RSS, 2024, KIT | [project](https://intuitive-robots.github.io/mdt_policy/), [code](https://github.com/YanjieZe/3D-Diffusion-Policy)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/pdf/2405.12213), ArXiv, 2024, UCB & Stanford & CMU & Google DeepMind | [project](https://octo-models.github.io/), [code](https://github.com/octo-models/octo)
- [Language Control Diffusion: Efficiently Scaling through Space, Time, and Tasks](https://arxiv.org/pdf/2210.15629), ICLR, 2024, Harvard | [project](https://lcd.eddie.win/), [code](https://github.com/ezhang7423/language-control-diffusion)
- [UniSim: Learning Interactive Real-World Simulators](https://openreview.net/pdf?id=sFyTZEqmUY), ICLR, 2024, UCB & Google DeepMind & MIT | [project](https://universal-simulator.github.io/unisim/)
</details>

<details><summary>2023</summary>

- [GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation](https://arxiv.org/abs/2312.13139), ArXiv, 2023.12, Bytedance | [project](https://gr1-manipulation.github.io/), [code](https://github.com/bytedance/GR-1)
- [RT-2: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2307.15818), CoRL, 2023, Google DeepMind |[project](https://robotics-transformer2.github.io/)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864), CoRL Workshop, 2023, Open X-Embodiment Collaboration | [project](https://robotics-transformer-x.github.io/), [code](https://github.com/google-deepmind/open_x_embodiment)
- [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://arxiv.org/abs/2307.05973), CoRL, 2023, Stanford | [project](https://voxposer.github.io/), [code](https://github.com/huangwl18/VoxPoser)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705), RSS, 2023, Stanford | [project](https://tonyzhaozh.github.io/aloha/), [code](https://github.com/tonyzhaozh/aloha)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137v4), RSS2023 & IJRR2024, Columbia | [project](https://diffusion-policy.cs.columbia.edu/), [code](https://github.com/columbia-ai-robotics/diffusion_policy)
- [HULC++: Grounding Language with Visual Affordances over Unstructured Data](https://arxiv.org/pdf/2210.01911.pdf), ICRA, 2023, University of Freiburg | [project](http://hulc2.cs.uni-freiburg.de/), [code](https://github.com/mees/hulc2)
- [Learning Universal Policies via Text-Guided Video Generation](https://arxiv.org/pdf/2302.00111.pdf), NeurIPS, 2023, MIT & Google Brain | [project](https://universal-policy.github.io/unipi/), [unofficial code](https://github.com/flow-diffusion/AVDC)
- [Learning to Act from Actionless Videos through Dense Correspondences](https://arxiv.org/abs/2310.08576), ArXiv, 2023, National Taiwan University | [project](https://flow-diffusion.github.io/), [code](https://github.com/flow-diffusion/AVDC)
</details>

<details><summary>2022</summary>

- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://say-can.github.io/assets/palm_saycan.pdf), CoRL, 2022, Google Robotics | [code](https://github.com/google-research/google-research/tree/master/saycan), [project](https://say-can.github.io/)
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817), ArXiv, 2022, Google Robotics | [code](https://github.com/google-research/robotics_transformer), [project](https://robotics-transformer1.github.io/)
</details>

<!-- ### Search Engine -->

### Scientific Discovery

<details><summary>2024</summary>

- [Agent Laboratory: Using LLM Agents as Research Assistants](https://huggingface.co/papers/2501.04227), ArXiv, 2025.1, AMD & JHU | [project](https://agentlaboratory.github.io/), [code](https://github.com/SamuelSchmidgall/AgentLaboratory)
</details>

### Multi-Agent

<details><summary>2024</summary>

- [WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis](https://arxiv.org/pdf/2412.03359), ArXiv, 2024.12, Alibaba. | [project](https://whoisspy.ai/), [code]()
- [Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks](https://arxiv.org/pdf/2411.04468), ArXiv, 2024, Microsoft. | [blog](https://aka.ms/magentic-one-blog), [code](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one)
- [SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents](https://github.com/David-Li0406/SMoA), ASU & MSU & KAUST & UT Austin. | [code](https://github.com/David-Li0406/SMoA)
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/pdf/2406.04692), ArXiv, 2024. 6, Together AI. | [code](https://github.com/togethercomputer/moa)
- [AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://openreview.net/pdf?id=EHg5GDnyq1), ICLR, 2024. THU & BUPT & Tecent | [code](https://github.com/OpenBMB/AgentVerse)
- [Metagpt: Meta programming for multi-agent collaborative framework](https://openreview.net/forum?id=VtmBAGCN7o), ICLR, 2024. DeepWisdom & KAUST & XMU & CUHKSZ & NJU & UoPenn & UCB | [code](https://github.com/geekan/MetaGPT)
- [ChatDev: Communicative Agents for Software Development](https://aclanthology.org/2024.acl-long.810), ACL, 2024. THU & BUPT & DUT etc. | [code](https://github.com/OpenBMB/ChatDev)
- [Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risk of Language Models](https://arxiv.org/abs/2408.08926), ArXiv, 2024.8 | [project](https://cybench.github.io), [code](https://github.com/andyzorigin/cybench)
- [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://arxiv.org/abs/2411.07763), ArXiv, 2024.11, HKU & Salesforce & Google Deepmind etc. | [project](https://spider2-sql.github.io/), [code](https://github.com/xlang-ai/Spider2)
- [Do as We Do, Not as You Think: the Conformity of Large Language Models](https://arxiv.org/abs/2410.12428), ArXiv, 2024.10, Cambridge | [project]()
- [Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence](https://arxiv.org/abs/2407.07061), ArXiv, 2024.7, THU & PKU & BUPT & Tencent | [project](https://openbmb.github.io/IoA/), [code](https://github.com/OpenBMB/IoA)
</details>

<details><summary>2023</summary>

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/pdf/2308.08155), ArXiv, 2023. MS & PSU & UoW | [code](https://github.com/microsoft/autogen)
- [CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society](https://proceedings.neurips.cc/paper_files/paper/2023/file/a3621ee907def47c1b952ade25c67698-Paper-Conference.pdf), NeurIPS, 2023, KAUST. | [project](https://www.camel-ai.org/), [code](https://github.com/camel-ai/camel)
</details>

<details><summary>2022</summary>

- [Diversity is All You Need: Learning Skills without a Reward Function](https://openreview.net/forum?id=SJx63jRqFm), ICLR, 2019, CMU & UCB & Google. | [project](https://sites.google.com/view/diayn/), [code](https://github.com/ben-eysenbach/sac)
- [Multi-Agent Reinforcement Learning is A Sequence Modeling Problem](https://proceedings.neurips.cc/paper_files/paper/2022/file/69413f87e5a34897cd010ca698097d0a-Paper-Conference.pdf), NeurIPS, 2022, SJTU | [project](https://sites.google.com/view/multi-agent-transformer), [code](https://github.com/PKU-MARL/Multi-Agent-Transformer)
- [Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=EcGGFkNTxdJ), ICLR, 2022, Oxford etc.
</details>

<details><summary>Before 2020</summary>

- [A actor-based architecture for customizing and controlling agent ensembles](https://agents.usask.ca/Papers/jamali-thati-agha-intelsys99.pdf), IEEE Intelligent Systems and their Applications, 1999, UIUC.
- [KQML as an agent communication language](https://dl.acm.org/doi/pdf/10.1145/191246.191322), CIKM, 1994.
</details>

### Others


<details><summary>2024</summary>

- [Haisor: Human-aware Indoor Scene Optimization via Deep Reinforcement Learning](https://dl.acm.org/doi/pdf/10.1145/3632947), ACM ToG, 2024, NVIDIA. | [project](https://research.nvidia.com/publication/2024-01_haisor-human-aware-indoor-scene-optimization-deep-reinforcement-learning)
</details>

<details><summary>2021</summary>

- [3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics](https://openaccess.thecvf.com/content/ICCV2021/papers/Fu_3D-FRONT_3D_Furnished_Rooms_With_layOuts_and_semaNTics_ICCV_2021_paper.pdf), ICCV, 2021, Alibaba & CAS & SFU. | [project](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), [code](https://github.com/droid-dataset/droid_policy_learning)
- [Multi Agent Reinforcement Learning of 3D Furniture Layout Simulation in Indoor Graphics Scenes](https://arxiv.org/pdf/2102.09137), ICLR SimDL Workshop, 2021, Longfor & IBM. | [code(partial)](https://github.com/CODE-SUBMIT/simulator2)
</details>

## Datasets and Benchmarks

### General AI Agent

- [Gaia: a benchmark for general ai assistants](https://openreview.net/pdf?id=fibxvahvs3), ICLR, 2024, Meta | [code](https://huggingface.co/gaia-benchmark), [leadboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)

### Robotics

<!-- summarization: https://www.xiaohongshu.com/explore/6749e8180000000002039d5c?xsec_token=AB6HX5PMGw8lloV2aCTzOkguQIsjy9kG3eX7TsZV9zwew=&xsec_source=pc_collect -->
- [DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945), ArXiv, 2024.3, DROID Dataset Team | [project](https://droid-dataset.github.io/)
- [RLBench](https://arxiv.org/abs/1909.12271), RA-L, 2020, ICL | [project](https://sites.google.com/view/rlbench), [code]((https://github.com/stepjam/RLBench))
- [CALVIN](https://arxiv.org/pdf/2112.03227.pdf), RA-L, 2022,University of Freiburg | [project](http://calvin.cs.uni-freiburg.de/), [code]((https://github.com/mees/calvin))
- [ManiSkill 1&2&3](https://arxiv.org/abs/2302.04659), ICLR, 2023, UCSD | [project](https://maniskill2.github.io/), [code](https://github.com/haosulab/ManiSkill)


## Other Resources

### Github Collections

- [Awesome AGI Agents](https://github.com/yzfly/Awesome-AGI-Agents)
- [Awesome Agents](https://github.com/kyrolabs/awesome-agents)
- [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents)
- [Awesome AI Agents](https://github.com/slavakurilyak/awesome-ai-agents)
- [Awesome LLM Powered Agent](https://github.com/hyp1231/awesome-llm-powered-agent)
- [Awesome AI Agents](https://github.com/Jenqyang/Awesome-AI-Agents)
- [Embodied_AI_Paper_List](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)
- [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [diffusion-literature-for-robotics](https://github.com/mbreuss/diffusion-literature-for-robotics)

### Tutorial

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

### Group

- [OpenAI](https://www.openai.com)
- [OpenBMB](https://github.com/OpenBMB)
- [DeepSeek](https://www.deepseek.com/)
- 

### Frameworks

#### Foundation Models

- [Llama Series](https://www.llama.com/)
- [Gemma Series](https://ai.google.dev/gemma)
- [Qwen Series](https://qwenlm.github.io/)
- [DeepSeek Series](https://www.deepseek.com/)
- [MiniMax Series](https://www.minimaxi.com/)


#### Multi-Agent

<!-- comparison: https://www.helicone.ai/blog/ai-agent-builders -->
- [openr](https://github.com/openreasoner/openr/)
- [swarm](https://github.com/openai/swarm)
- [autogen](https://github.com/microsoft/autogen)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [LangChain](https://github.com/langchain-ai/langchain)
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
- [AgentGPT](https://github.com/reworkd/AgentGPT)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)
- [AgentVerse](https://github.com/OpenBMB/AgentVerse)
- [Crew AI](https://github.com/crewAIInc/crewAI)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CAMEL](https://github.com/camel-ai/camel)

### Links

- [RSS2024 Papers](https://roboticsconference.org/2024/program/papers/)
- [Papercopilot](https://papercopilot.com/statistics/#robotics)


### Star History
[![Star History Chart](https://api.star-history.com/svg?repos=weleen/awesome-agent&type=Timeline)](https://star-history.com/#weleen/awesome-agent&Timeline)

