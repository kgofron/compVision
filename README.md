# compVision #
# WARNING: TRANSITION TO OPENCV 3.1 #
This is a computer vision wrapper module along with pratical applications at the NSLS-2 facility at Brookhaven National Lab.
This work was done at the Inelastic X-ray Scattering (IXS) 10-ID Beamline At the National Synchrotron Light Source - 2 at BNL. Upon Integration with EPICS, these programs will be more useful.
Summer 2016

### Author ###
* William Watson

### Acknowledgments ###
* Kaz Gofron - My Mentor at BNL. 
* NSLS-2 IXS (10-ID) Beamline Staff

### Dependencies ###
* `OpenCV` - for Computer Vision
* `EPICS` - for Server and Database Integration
* `Numpy` - for array functions and structures
* `Matplotlib` - for displaying images and spectrums that cannot be handled by OpenCV

### Files ###
* `cvlib.py` - Wrapper CV Module
* `LICENSE` - MIT License for this Project
* `README.md` - Markdown Readme File
* `fire.sh` - Git Shell Script
* `updateLib.sh` - Shell Script to update cvlib.py in all subfolders.
* `cvlib.html` - HTML Documentation File
* `genTreeTable.sh` - Generates Tree File
* `generateDocs.sh` - Generates Documentation for cvlib
* `gitlog.txt` - Git Log for Project
* `showDocs.sh` - Displays Documentation in console
* `tree.txt` - Tree Diagram for Git Repo
* `wordcount.sh` - Counts Total Lines in all files and code files
* `cvlib.pyc` - Compiled Python Library

### Folders ###
* `CVLibrary` - Contains a copy of cvlib.py as a standalone
* `Crystals` - Sub application used for crystal detection and X-Ray streak analysis
* `IXS` - Sub programs for cv related applications to IXS Cameras
* `Merlin` - Programs written for Merlin Detector Image Analysis
* `OnAxisImg` - Image Analysis for On Axis Camera at IXS
* `RawImages` - Programs Designed to pass jusdgement on Pin and Gripper Placement
* `Showcase` - Programs designed to show to other scientists the value of integrating cv to their beamlines
* `tests` - Testing Files and Programs written just for fun, with wacky images
