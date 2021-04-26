# 🍇 GOMU
## Installation
### Environment-settings
- python 3.8
- pytorch 1.6.0
- transformers 4.6.0.dev0

**You can install with following steps**
1. `conda create -n gomu python==3.8`
2. `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`
3. `cd text-classification/`
4. `pip install -r requirements.txt`
5. `pip install git+https://github.com/huggingface/transformers #install with source` 

## Run
### Finetune pipeline
**RTE**  
`./scripts/run_rte.sh`  
**MNLI**  
`./scripts/run_mnli.sh`  

### CrowS-Pairs bias measurement

### StereoSet bias measurement




## TMIs
### About 'GOMU'
The word "GOMU" is Korean word meaning 'rubber', which sounds like 'lover'. It is also related to a renowned album by The Beatles, <rubber soul>. We believe that the world only changes by people who never stops loving something.
 
### Committing rules

- build: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)  
- ci: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)  
- docs: Documentation only changes  
- feat: A new feature  
- fix: A bug fix  
- perf: A code change that improves performance  
- refactor: A code change that neither fixes a bug nor adds a feature  
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)  
- test: Adding missing tests or correcting existing tests  

