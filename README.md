# PyTorchEngine
**
Finished four main step before using this framework**

## 1. engine/dataset
1. define your own dataset
2. use tools/test_cfg.py to test if it works as expected
   
## 2. engine/models
1. create your own model pipeline
2. use tools/test_cfg.py to test if it works as expected

## 3. engine/loss
1. create your training loss
2. use tools/test_cfg.py to test if it works as expected
3. copy it to engine/engine.py - compute_loss()

## 4. engine/engine.py
1. After finished the previous steps, you are almost ready
2. setup your config/myconfig.yaml file
3. run train.py to see if it works properly
4. define your evaluation metric 
5. use your evaluation metric result to save your model checkpoint