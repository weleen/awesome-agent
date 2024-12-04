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

## Table of Contents

- [Awesome Agent](#awesome-agent)
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

### Foundation Model (LLM, VLM, etc.)

<details><summary>2024</summary>

- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/pdf/2408.03326), ArXiv, 2024.8, NTU & Bytedance | [project](https://llava-vl.github.io/blog/2024-08-05-llava-onevision), [code](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation](https://arxiv.org/pdf/2409.04429), ArXiv, 2024.9, THU & MIT & NVIDIA & UCB & UCSD. | [project](https://hanlab.mit.edu/projects/vila-u), [code](https://github.com/mit-han-lab/vila-u)
- [Qwen2 & QwQ: Reflect Deeply on the Boundaries of the Unknown](https://arxiv.org/abs/2407.10671), ArXiv, 2024.7 & 2024.11, Qwen Team at Alibaba | [model](https://huggingface.co/Qwen/QwQ-32B-Preview), [project](https://huggingface.co/Qwen), [blog](https://qwenlm.github.io/blog/qwq-32b-preview/)
- [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/abs/2411.14405), ArXiv, 2024.11, AIDC-AI at Alibaba | [model](https://huggingface.co/AIDC-AI/Marco-o1), [code](https://github.com/AIDC-AI/Marco-o1)
- [O1 Journey: O1 Replication Journey – Part 2: Surpassing O1-preview through Simple Distillation Big Progress or Bitter Lesson?](https://arxiv.org/abs/2411.16489), ArXiv, 2024.11, GAIR SJTU | [code](https://github.com/GAIR-NLP/O1-Journey)
- [Gemma: Open models based on gemini research and technology](https://arxiv.org/abs/2403.08295), ArXiv, 2024.3, Google | [project](https://blog.google/technology/developers/gemma-open-models/)
- [Minicpm: Unveiling the potential of small language models with scalable training strategies](https://arxiv.org/pdf/2404.06395), CoLM, 2024, OpenBMB. | [blog](https://openbmb.vercel.app/?category=Blog), [code](https://github.com/OpenBMB/MiniCPM)
 </details>

### Robotics

<details><summary>2024</summary>

- [ARMOR：Egocentric Perception for Humanoid Robot Collision Avoidance and Motion Planning](https://arxiv.org/abs/2412.00396), ArXiv, 2024.12. CMU & Apple | [code]
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

### Multi-Agent

<details><summary>2024</summary>

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
</details>

<details><summary>2022</summary>

- [Multi-Agent Reinforcement Learning is A Sequence Modeling Problem](https://proceedings.neurips.cc/paper_files/paper/2022/file/69413f87e5a34897cd010ca698097d0a-Paper-Conference.pdf), NeurIPS, 2022, SJTU | [project](https://sites.google.com/view/multi-agent-transformer), [code](https://github.com/PKU-MARL/Multi-Agent-Transformer)
- [Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=EcGGFkNTxdJ), ICLR, 2022, Oxford etc.
</details>

<details><summary>Before 2020</summary>

- [A actor-based architecture for customizing and controlling agent ensembles](https://agents.usask.ca/Papers/jamali-thati-agha-intelsys99.pdf), IEEE Intelligent Systems and their Applications, 1999, UIUC.
- [KQML as an agent communication language](https://dl.acm.org/doi/pdf/10.1145/191246.191322), CIKM, 1994.
</details>

### Others

## Datasets and Benchmarks

### General AI Agent

- [Gaia: a benchmark for general ai assistants](https://openreview.net/pdf?id=fibxvahvs3), ICLR, 2024, Meta | [code](https://huggingface.co/gaia-benchmark), [leadboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)

### Robotics

<!-- summarization: https://www.xiaohongshu.com/explore/6749e8180000000002039d5c?xsec_token=AB6HX5PMGw8lloV2aCTzOkguQIsjy9kG3eX7TsZV9zwew=&xsec_source=pc_collect -->
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

### Tutorial

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

### Group

- [OpenAI](https://www.openai.com)
- [OpenBMB](https://github.com/OpenBMB)

### Frameworks

#### Foundation Models

- [Qwen Series](https://qwenlm.github.io/)
- [Llama Series](https://www.llama.com/)
- [Gemma Series](https://ai.google.dev/gemma)


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