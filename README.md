
# Tale Hisdal Sandberg - Gesture Control of Quadruped Robots

Investigating user acceptance and technology barriers to gesture control of mobile quadruped robots in real world use cases.



The data sets are numpy arrays of OpenPose vector output, which was used to train the model used in live user experiments.
The files names are coded, e.g. g03 is one gesture while g04 is a different gesture. 


Be aware that the code names do not quite match up to the code names used in the master thesis document.

- g03 is g13 left arm up in the thesis
- g04 is g012 right arm up in the thesis
- g07 is g07 t-pose in the thesis
- g08 is g17 left arm V in the thesis
- g09 is g14 both arms up in the thesis
- g10 is g11 squat in the thesis
- g11 is neutral in the thesis
- g13 is g08 right arm V in the thesis




Usage of training code:

python3 training_code.py

Usage of live code with robot:

python3 run_live_code.py

(Make sure to have the correct directory path for the trained model weights. Also OpenPose might not work unless the python script is placed within the OpenPose python api tutorial directory which comes with the installation of OpenPose).


Requirements:

python3, numpy, math, keras and tensorflow, OpenPose GPU version, Spot Boston Dynamics api packages.


Instalment of OpenPose can be found here:
https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_installation_0_index.html
