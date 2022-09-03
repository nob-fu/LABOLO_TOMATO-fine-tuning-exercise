# LABOLO_TOMATO fine-tuning exercise
# プロダクト開発演習

## １．テーマ選定
- 目的： 農作物生産作業の効率化、省人化
- 背景： 日本の食料自給率低下、農業従事者の高齢化の一方で、人工知能、高速通信（５G）、ドローン、ロボティクスなどICT技術拡大を背景として、スマート農業への期待が高まっている。
- 採用手段： 農作物の生育状況を画像から判定する。今回の演習では、トマト画像から、トマト果実、1個ずつの成熟度をクラス分類する。このため、セグメンテーションの新分野となる「オブジェクト・インスタンス・セグメンテーション」の適用を行う。静止画、動画どちらでも判断できることを要件のひとつとする。
- 期待効果： 生育状況の把握、収穫時期・地点の推定、収穫量予測、収穫作業の自動化のシステム／プロセスに効果的な手段の提供

**【テーマ】トマト画像（静止画、動画）に対するインスタンス・セグメンテーションを実施し、画像中、果実の位置を個別にマスクし、その成熟度（成熟、中間、未成熟）クラスを判定する**

## ２．参考情報の収集
 1. 公開されている学習用ラベル済みトマト画像と学習済みモデル
株式会社LABORO.AIがトマト画像・物体検出データセット『Laboro Tomato』を公開（2020-07-14）。  
  (https://laboro.ai/activity/column/engineer/laboro-tomato/)  
  (https://github.com/laboroai/LaboroTomato)  
 2. インスタンス・セグメンテーションをサポートするPyTorch対応ツールキット
PyTorch向けの物体検出ライブラリーとしては、Detectron2（Meta社）、MMDetection（OpenMMLab）などがある。なおLaboro TomatoではMMDetectionを使い、学習済みパラメタ（checkpoint)とそのMask R-CNNモデルのconfig情報、アノテーション済みdataset（MS COCO形式）を提供。  
  (https://mmdetection.readthedocs.io/en/latest/)  
  (https://github.com/open-mmlab/mmdetection)  
 3. インスタンス・セグメンテーションのアルゴリズム動向
物体検出モデル（分類：cls、回帰：BBox）としてはFaster R-CNN, YOLO, SSDなど、またインスタンス・セグメンテーション（cls, BBox, & mask）としては、Mask R-CNN(2017), YOLACT(2019), SOLO(2020)などがある。  
 4. 無料で利用可能なフリーライセンス画像ソース
ロイヤルティフリーで利用可能な画像の提供サイトがあり、一定の条件下で学習用画像としても利用できるものもある。今回、検証用画像の一部に「写真AC」（ https://www.photo-ac.com/ ）の画像を利用する。（個人での商用利用可、加工可）  
Windowsで利用できるアノテーション用ツールとしては、labelme（power shellから起動）、coco-annotator（docker コンテナ）などがある。（今回は使用せず）  

- Mask R-CNN：https://arxiv.org/abs/1703.06870 (He et al., 2018)  
特徴：B-Box高精度、多機能（姿勢検出に拡張可能）、～5 fps  
- YOLACT：https://arxiv.org/abs/1904.02689 (Bolya et al., 2019)  
特徴：判定（inference）の高速性、B-Box回帰精度はやや低  
29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan X  
- SOLO：https://arxiv.org/abs/1912.04488 (Wang et al., 2020)  
 SOLOv2：https://arxiv.org/abs/2003.10152 (Wang et al., 2020)    
特徴：高精度かつ高速 

![image2_2](https://github.com/nob-fu/LABOLO_TOMATO-fine-tuning-exercise/images/image2_2.png)
 
