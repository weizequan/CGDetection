### CGDetection Dataset

Download two natural image datasets.
- `RAISE`: 8156 raw images from the [Raw Images Dataset](http://loki.disi.unitn.it/RAISE/download.html).
- `VISION`: VISION dataset from the [LESC](https://lesc.dinfo.unifi.it/en/datasets). 

To construct NI(CG) dataset, you need reference the detailed description in Section 3 (Datasets) of our paper. In addition, the corresponding data splits are reported in TrainValTestSplit/NI(CG)_*.txt.

CG Dataset preparation.
- `Artlantis`: download the images using Artlantis/Artlantis_URL.txt, and recommend compressing them.
- `Autodesk`: download the images from [Autodesk](https://drive.google.com/file/d/1rTB0OyVPXe1GvRnBJ_zM1_1Jxj5hoqAT/view?usp=sharing).
- `Corona`: download the images using Corona/Corona_URL.txt, and then run Corona/imageCrop.py to remove the logo information.
- `VRay`: follow the process in VRay/ReadMe.txt, and images in learnVRay can be download from [learnVRay](https://drive.google.com/file/d/1EnJ-C2tZGG6IMQLKVv3MrcFCPecXIZJE/view?usp=sharing).

## Citation
```
@inproceedings{nguyen_raise_2015,
 author = {Dang-Nguyen, Duc-Tien and Pasquini, Cecilia and Conotter, Valentina and Boato, Giulia},
 title = {{RAISE}: {A} Raw Images Dataset for Digital Image Forensics},
 booktitle = {Proceedings of the ACM Multimedia Systems Conference},
 year = {2015},
 pages = {219--224}
}

@Article{shullani_vision_2017,
	author="Shullani, Dasara
	and Fontani, Marco
	and Iuliani, Massimo
	and Shaya, Omar Al
	and Piva, Alessandro",
	title="{VISION}: a video and image dataset for source identification",
	journal="EURASIP Journal on Information Security",
	year="2017",
	volume="2017",
	pages="15",
	number="1"
}
```
