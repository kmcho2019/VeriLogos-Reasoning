# 😎 VeriLogos 😎

> This repository includes the scripts utilized in the study titled "Improving LLM-based Verilog Code Generation with Data Augmentation and RL (DATE25)".

![Image](https://github.com/user-attachments/assets/68b5c801-1a32-42a0-8759-2ff2f79a016c)

(↑ Overall Framework)

## ⭐ Main Feature
### AST-based Data Augmentation  
- Represent Verilog code into an abstract syntax tree (AST), perform node-level augmentation, and generate synthetic Verilog code
### Output-relevant Code Masking
- Represent Verilog code into an AST, identity variables related to an output signal, and generate a prompt by masking the corresponding code sections
### Reinforcement Learning with Tool Feedback (RLTF) 
- Fine-tune the model by integrating RL and EDA Tools. EDA tools evaluate the model-generated Verilog code and compute rewards based on syntax and functionality

## 💻 Getting Started
### Docker Setting 
```
# Docker image download 
docker pull 97kjmin/verilogos:1.0

# Docker container build 
docker run -it --name <name> --gpus '"device=<0,1,2...>"' --net host -v <path/to/local>:<path/to/docker> --shm-size='16gb' 97kjmin/verilogos:1.0 bash
```
### Git 
```
git clone https://github.com/97kjmin/VeriLogos.git
```
### Accelerate Setting 
```
accelerate config --config_file <path/to/config/my_config.yaml> 
```
### Run 
```
accelerate launch --config_file <path/to/config/my_config.yaml> main.py ...
```
(Please refer to the file in the 'run_script' directory)

## 📖 Model & Dataset 
### Model
```
https://huggingface.co/97kjmin/VeriLogos
```
### Augmented Dataset 
```
https://huggingface.co/datasets/97kjmin/VeriLogos_Augmented_Dataset
```
(Since data scraped from websites may involve copyright issues, we are now releasing 9,881 code datasets that have been augmented and anonymized)

## ⭐ Note
1. To run RLTF, both iVerilog and Synopsys Formality are required. Each must be installed separately, and Synopsys Formality additionally requires a license, so please bear this in mind.
2. There may have been unexpected issues during the process of organizing the code for upload. If you encounter any problems while running it, please let us know.
3. Thank you and have a good day :)  

## :open_file_folder: Project Structure

```markdown
VeriLogos
├── verilogos
│   ├── augmentator
│   ├── evaluator
│   ├── generator
│   ├── trainer
│   └── utils
├── ref
├── run_script
└── main.py

```

## 👨‍👩‍👧‍👦 Developer
*  **Kyungjun Min** ([microhumanis](https://github.com/97kjmin))
*  **Seonghyeon Park** ([shypark98](https://github.com/shypark98))
*  **Hyeonwoo Park** ([kangnam4123](https://github.com/kangnam4123))
*  **Jinoh Cho** ([Jinoh-Cho](jinoma0927@gmail.com))
