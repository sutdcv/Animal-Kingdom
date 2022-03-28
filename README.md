# Animal Kingdom
![Image](https://github.com/SUTDCV/Animal-Kingdom/blob/master/image/header.png)
This is the official repository for 
<br/>**[[CVPR2022] Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding]()**
<br/>**Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, Jun Liu**
<br/>*Information Systems Technology and Design, Singapore University of Technology and Design, Singapore*

## Abstract
Understanding animals' behaviors is significant for a wide range of applications. However, existing animal behavior datasets have limitations in multiple aspects, including limited numbers of animal classes, data samples and provided tasks, and also limited variations in environmental conditions and viewpoints. To address these limitations, we create a large and diverse dataset, Animal Kingdom, that provides multiple annotated tasks to enable a more thorough understanding of natural animal behaviors. The wild animal footages used in our dataset record different times of the day in extensive range of environments containing variations in backgrounds, viewpoints, illumination and weather conditions. More specifically, our dataset contains 50 hours of annotated videos to localize relevant animal behavior segments in long videos for the video grounding task, 30K video sequences for the fine-grained multi-label action recognition task, and 33K frames for the pose estimation task, which correspond to a diverse range of animals with 850 species across 6 major animal classes. Such a challenging and comprehensive dataset shall be able to facilitate the community to develop, adapt, and evaluate various types of advanced methods for animal behavior analysis. Moreover, we propose a Collaborative Action Recognition (CARe) model that learns general and specific features for action recognition with unseen new animals. This method achieves promising performance in our experiments.

<!-- # Citation -->

# Dataset
[Download dataset here](https://forms.office.com/r/WCtC0FRWpA)

## Video Grounding
![Image](https://github.com/SUTDCV/Animal-Kingdom/blob/master/image/eg_vg.png)

Table 4: Results of video grounding
<!-- |        | Recall@1 |          |          |          | mean IoU |
| :----: | :------: | :------: | :------: | :------: | :------: |
| Method | IoU=0\.1 | IoU=0\.3 | IoU=0\.5 | IoU=0\.7 |          |
| LGI    | 50\.84   | 33\.51   | 19\.74   | 8\.94    | 22\.90   |
| VSLNet | 53\.59   | 33\.74   | 20\.83   | 12\.22   | 25\.02   | -->

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Results of video grounding
	</caption>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<td colspan="4" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Recall@1
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			mean IoU
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Method
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU=0.1
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU=0.3
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU=0.5
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			IoU=0.7
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			LGI
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			50.84
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			33.51
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			19.74
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			8.94
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			22.90
	<tr>
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			VSLNet
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			53.59
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			33.74
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			20.83
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			12.22
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			25.02
</table>

## Action Recognition
![Image](https://github.com/SUTDCV/Animal-Kingdom/blob/master/image/eg_ar.png)
Table 2. Results of action recognition
<!-- |                               | mAP     |        |        |        |
| :---------------------------: | :------ | :----: | :----- | :----: |
| Method                        | overall | head   | middle | tail   |
| Baseline (Cross Entropy Loss) |         |        |        |        |
| I3D                           | 16\.48  | 46\.39 | 20\.68 | 12\.28 |
| SlowFast                      | 20\.46  | 54\.52 | 27\.68 | 15\.07 |
| X3D                           | 25\.25  | 60\.33 | 36\.19 | 18\.83 |
| Focal Loss                    |         |        |        |        |
| I3D                           | 26\.49  | 64\.72 | 40\.18 | 19\.07 |
| SlowFast                      | 24\.74  | 60\.72 | 34\.59 | 18\.51 |
| X3D                           | 28\.85  | 64\.44 | 39\.72 | 22\.41 |
| LDAM-DRW                      |         |        |        |        |
| I3D                           | 22\.40  | 53\.26 | 27\.73 | 17\.82 |
| SlowFast                      | 22\.65  | 50\.02 | 29\.23 | 17\.61 |
| X3D                           | 30\.54  | 62\.46 | 39\.48 | 24\.96 |
| EQL                           |         |        |        |        |
| I3D                           | 24\.85  | 60\.63 | 35\.36 | 18\.47 |
| SlowFast                      | 24\.41  | 59\.70 | 34\.99 | 18\.07 |
| X3D                           | 30\.55  | 63\.33 | 38\.62 | 25\.09 | -->

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Results of action recognition
	</caption>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<td colspan="4" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			mAP
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Method
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			overall
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			head
		<td style="border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			middle
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			tail
	<tr>
		<td colspan="5" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Baseline (Cross Entropy Loss)
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			I3D
		<td style="padding-right: 3pt; padding-left: 3pt;">
			16.48
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			46.39
		<td style="padding-right: 3pt; padding-left: 3pt;">
			20.68
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			12.28
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SlowFast
		<td style="padding-right: 3pt; padding-left: 3pt;">
			20.46
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			54.52
		<td style="padding-right: 3pt; padding-left: 3pt;">
			27.68
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			15.07
	<tr>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			X3D
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			25.25
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			60.33
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			36.19
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			18.83
	<tr>
		<td colspan="5" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Focal Loss
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			I3D
		<td style="padding-right: 3pt; padding-left: 3pt;">
			26.49
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			64.72
		<td style="padding-right: 3pt; padding-left: 3pt;">
			40.18
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			19.07
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SlowFast
		<td style="padding-right: 3pt; padding-left: 3pt;">
			24.74
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			60.72
		<td style="padding-right: 3pt; padding-left: 3pt;">
			34.59
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			18.51
	<tr>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			X3D
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			28.85
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			64.44
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			39.72
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			22.41
	<tr>
		<td colspan="5" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			LDAM-DRW
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			I3D
		<td style="padding-right: 3pt; padding-left: 3pt;">
			22.40
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			53.26
		<td style="padding-right: 3pt; padding-left: 3pt;">
			27.73
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			17.82
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SlowFast
		<td style="padding-right: 3pt; padding-left: 3pt;">
			22.65
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			50.02
		<td style="padding-right: 3pt; padding-left: 3pt;">
			29.23
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			17.61
	<tr>
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			X3D
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			30.54
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			62.46
		<td style="border-bottom: 0px solid rgb(0, 0, 0); padding-right: 3pt; padding-left: 3pt;">
			39.48
		<td style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			24.96
	<tr>
		<td colspan="5" style="border-bottom: 0px solid rgb(0, 0, 0); text-align: center; padding-right: 3pt; padding-left: 3pt;">
			EQL
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			I3D
		<td style="padding-right: 3pt; padding-left: 3pt;">
			24.85
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			60.63
		<td style="padding-right: 3pt; padding-left: 3pt;">
			35.36
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			18.47
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			SlowFast
		<td style="padding-right: 3pt; padding-left: 3pt;">
			24.41
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			59.70
		<td style="padding-right: 3pt; padding-left: 3pt;">
			34.99
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			18.07
	<tr>
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			X3D
		<td style="border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			30.55
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			63.33
		<td style="border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			38.62
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			25.09
</table>

Table 3: Results of action recognition of unseen animals
<!-- | Method                                 | Accuracy (\\%)      |
| :------------------------------------: | :-----------------: |
| Episodic-DG                            | 34\.0               |
| Mixup                                  | 36\.2               |
| CARe without specific feature          | 27\.3               |
| CARe without general feature           | 38\.2               |
| CARe without spatially-aware weighting | 37\.1               |
| CARe (Our full model)                  | 39\.7               | -->
<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Results of action recognition of unseen animals
	</caption>
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Method
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Accuracy (%)
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Episodic-DG
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			34.0
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Mixup
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			36.2
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			CARe without specific feature
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			27.3
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			CARe without general feature
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			38.2
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			CARe without spatially-aware weighting
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			37.1
	<tr>
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			CARe (Our full model)
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			39.7
</table>

 
## Pose Estimation
![Image](https://github.com/SUTDCV/Animal-Kingdom/blob/master/image/eg_ar.png)
Table 5. Results of pose estimation
<!-- |   <br>                     |                   | PCK@0\.05 |            |
| :------------------------: | :---------------: | :-------: | :--------: |
| Protocol                   | Description       | HRNet     | HRNet-DARK |
| Protocol 1                 | All               | 66\.06    | 66\.57     |
| Protocol 2                 | Leave-*k*-out     | 39\.30    | 40\.28     |
| Protocol 3                 | Mammals           | 61\.59    | 62\.50     |
|                            | Amphibians        | 56\.74    | 57\.85     |
|                            | Reptiles          | 56\.06    | 57\.06     |
|                            | Birds             | 77\.35    | 77\.41     |
|                            | Fishes            | 68\.25    | 69\.96     | -->

<table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<caption>
		Results of pose estimation
	</caption>
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<br>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
		<td colspan="2" style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			PCK@0.05
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Protocol
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Description
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			HRNet
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			HRNet-DARK
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Protocol 1
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			All
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			66.06
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			66.57
	<tr>
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Protocol 2
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Leave-<i>k</i>-out
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			39.30
		<td style="border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			40.28
	<tr>
		<td rowspan="5" style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			Protocol 3
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Mammals
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			61.59
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			62.50
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Amphibians
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			56.74
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			57.85
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Reptiles
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			56.06
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			57.06
	<tr>
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Birds
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			77.35
		<td style="text-align: center; padding-right: 3pt; padding-left: 3pt;">
			77.41
	<tr>
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			Fishes
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			68.25
		<td style="text-align: center; border-bottom: 2px solid black; padding-right: 3pt; padding-left: 3pt;">
			69.96
</table>

# Acknowledgement and Contributors
This project is supported by AI Singapore (AISG-100E-2020-065) and SUTD Startup Research Grant.

We would like to thank the following contributors for working on the annotations and conducting the quality checks for video grounding, action recognition and pose estimation. 
1.	Ang Yu Jie
2.	Ann Mary Alen
3.	Cheong Kah Yen Kelly
4.	Foo Lin Geng
5.	Gong Jia
6.	Heng Jia Ming
7.	Javier Heng Tze Jian
8.	Javin Eng Hee Pin
9.	Jignesh Sanjay Motwani
10.	Li Meixuan
11.	Li Tianjiao
12.	Liang Junyi
13.	Loy Xing Jun
14.	Nicholas Gandhi Peradidjaya
15.	Song Xulin
16.	Tian Shengjing
17.	Wang Yanbao
18.	Xiang Siqi
19.	Xu Li
