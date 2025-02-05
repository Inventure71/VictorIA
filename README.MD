# Guide + Things to remember
**FIRST** 
- try: Install the requirements.txt file using the following command:<br>
```pip install -r requirements.txt```
- _RobotExperimental.py_ is supposed to be run in the ubuntu environment, modify the IP if necessary

**IF SOMETHING DOESN'T WORK** <br>
_Possible issues:_
- I installed SAM2 into the directory before the project (../SAM2)
  - To do the same:
    - Start terminal in project directory
    - ```cd ..```
    - Clone the SAM2 repository from the following link:
    - ```git clone https://github.com/facebookresearch/sam2.git && cd sam2```
    - ```pip install -e .``` (_**optional**_) (Try **without** this because after this step tensorflow might give issues, fix is below)
    - ```cd checkpoints```
    - ```./download_ckpts.sh```
    - Close the terminal instance / go back to project directory


- Tensorflow is really critical, install 1.12
  - Use the following command to install tensorflow 1.12, 
    - on mac use:<br>
```conda install tensorflow-macos==1.12```
    - on windows use (if not working try to install the CPU / GPU versions they are different):<br>
```conda install tensorflow==1.12.0``` 
    - You also need to adapt numpy to a version compatible with this tensorflow and SAM2, using the following command:<br>
```pip install "numpy<2.0.0"```


- Teachable machine model is not working because it is for older versions of keras
  - Use the script _utils/FIX-Teachable-Machine.py_ 
  - It will fix and substitute the old keras model with the new one