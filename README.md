# compVision #
This is a computer vision wrapper module along with pratical applications at the NSLS-2 facility at Brookhaven National Lab.
This work was done at the Inelastic X-ray Scattering (IXS) 10-ID Beamline At the National Synchrotron Light Source - 2 at BNL. Upon Integration with EPICS, these programs will be more useful.
Summer 2016

### Author ###
* William Watson

### Acknowledgments ###
* Kaz Gofron - My Mentor at BNL. 
* NSLS-2 IXS Beamline - Yong Cai, Alessandro Cunsolo, Alexey Suvorov
* NSLS-2 AMX Beamline - Jean Jakoncic
* NSLS-2 XPD Beamline - Sanjit Ghose

### Dependencies ###
* `OpenCV` - for Computer Vision
* `EPICS` - for Server and Database Integration
* `Numpy` - for array functions and structures
* `Matplotlib` - for displaying images and spectrums that cannot be handled by OpenCV

### Files ###
* `cvlib.py` - Wrapper CV Module
* `cvlibNoEpics.py` - CV Module Without EPICS, useful for just cv without Beamline integration
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
* `AMX` - Contains Sub Folders for Pin/Gripper and Loop Programs
* `EPICS` - Contains Files Associated to EPICS Server
* `IXS` - Subprograms for Merlin, BPM, Camera Analysis for IXS
* `Poster` - Subprograms made for posters by Kaz Gofron
* `Showcase` - Testing Programs for Demos
* `XPD` - Programs and Images related to X-Ray Diffractions for XPD
* `tests` - Testing files and concepts
