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
ロイヤルティフリーで利用可能な画像の提供サイトがあり、一定の条件下で学習用画像としても利用できるものもある。今回、検証用画像の一部に [写真AC](https://www.photo-ac.com/) の画像を利用する。（個人での商用利用可、加工可）  
Windowsで利用できるアノテーション用ツールとしては、labelme（power shellから起動）、coco-annotator（docker コンテナ）などがある。（今回は使用せず）  

- Mask R-CNN：https://arxiv.org/abs/1703.06870 (He et al., 2018)  
特徴：B-Box高精度、多機能（姿勢検出に拡張可能）、～5 fps  
- YOLACT：https://arxiv.org/abs/1904.02689 (Bolya et al., 2019)  
特徴：判定（inference）の高速性、B-Box回帰精度はやや低  
29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan X  
- SOLO：https://arxiv.org/abs/1912.04488 (Wang et al., 2020)  
 SOLOv2：https://arxiv.org/abs/2003.10152 (Wang et al., 2020)    
特徴：高精度かつ高速 

![image2_1](https://github.com/nob-fu/LABOLO_TOMATO-fine-tuning-exercise/blob/main/images/image2_1.png)  
![image2_2](https://github.com/nob-fu/LABOLO_TOMATO-fine-tuning-exercise/blob/main/images/image2_2.png)  

## ３．実施方針の決定
a. Laboro Tomato Datasetを利用し、MMDtectionフレームワーク上でMask R-CNNモデルの検証を行う  
b. YOLACTモデルをファインチューニングさせ、Mask R-CNNモデルとの比較検証を行う  
c. 学習(train)、評価(test)用データはLaboro Tomatoを使い、MMDetectionフレームワークで実行する  
d. 上記とは別に、検証用データとしてスマートフォン撮影動画・静止画、ロイヤルティフリー画像を使用する  
e. 本課題中では、検証用データにはアノテーションを行わず、出力された推定結果を目視確認して、定性的な評価と考察のみ行う  
f. 以下のステップで実施する  
  1) MMDetectionフレームワークの動作確認、取扱い習得
  公開tutorialによるMMDetectionの実行環境構築、動作確認
  (https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb)
  2) Laboro Tomato dataset, pretrained modelの検証  
  実行環境を再現、test dataによる正当性の確認（validation）、新たに準備したデータによる検証（verification）  
  3) Laboro Tomato datasetを使い、YOLACT modelに切り替えての検証  
  実行環境構築（COCO2017学習済みconfig）、train dataによるファインチューニング、test dataによる評価・計測（evaluation）、新たに準備したデータによる検証（verification）

## ４．データセット準備 
Laboro Tomatoデータの内訳
~~~
name: tomato_mixed    # 学習用：643、評価用：161のjpegファイル
images: 643 train, 161 test
cls_num: 6
cls_names: b_fully_ripened, b_half_ripened, b_green,
           l_fully_ripened, l_half_ripened, l_green
# トマト：成熟、中間、未成熟、ミニトマト：成熟、中間、未成熟の６クラス
total_bboxes: train[7781], test[1,996]
bboxes_per_class:
    *Train: b_fully_ripened[348], b_half_ripened[520], b_green[1467],
            l_fully_ripened[982], l_half_ripened[797], l_green[3667]
    *Test:  b_fully_ripened[72], b_half_ripened[116], b_green[387],
            l_fully_ripened[269], l_half_ripened[223], l_green[929]
~~~

datasetディレクトリ構造
~~~
'''
data  
├── laboro_tomato  
    ├── annotations ### COCO annotation  
    │ ├── train.json, test.json  
    ├── train ### train image datasets, 643 jpegファイル  
    ├── test ### test image datasets, 161 jpegファイル  
### image_resolutions: 3024x4032, 3120x4160の2種混在  
'''
~~~

COCO annotation 形式jsonファイル構造：
~~~
{
"images": [image],
"annotations": [annotation],
"categories": [category]
}
image = {
"id": int,
"width": int,
"height": int,
"file_name": str,
}
annotation = {
"id": int,
"image_id": int,
"category_id": int,
"segmentation": RLE or [polygon],
"area": float,
"bbox": [x,y,width,height],
"iscrowd": 0 or 1,
}
categories = [{
"id": int,
"name": str,
"supercategory": str,
}]
~~~

data/eval_tomato フォルダ内評価用静止画ファイル（アノテーションなし）
| File名      |サイズ     |記事                  | 
|:------------:|---------:|:---------------------|
|eval_001.jpg|640×480 |ミニトマト、水滴付き|1
|eval_002.jpg|640×480 |トマト、水滴付き|
|eval_003.jpg|640×480 |ミニトマト|
|eval_004.jpg|640×427 |トマト|
|eval_005.jpg|427×640 |トマト|
|eval_006.jpg|1920×1280 |ミニトマト、箱入り、多品種|
|eval_007.jpg|640×640 |ミニトマト|
|eval_008.jpg|640×360 |ミニトマト、動画ファイルからカットした為ピンボケ気味|
|eval_009.jpg|640×427 |リンゴ、木成り|
|eval_010.jpg|427×640 |リンゴ、木成り|

data/video_tomato フォルダ内評価用動画ファイル（アノテーションなし）  
| File名      |サイズ     |記事                  | 
|:------------:|---------:|:---------------------|
|tomato1.mp4|960×540 |00:00:08, 30.00 flm/s|
|tomato2.mp4|640×480 |00:00:03, 29.97 flm/s|
|tomato3.mp4|640×480 |00:00:58, 29.97 flm/s|
スマートフォン撮影したオリジナルファイルは、サイズ：
640×360、フレームレート：24.00flm/sで記録されているので、
PC上の動画編集ソフトにて編集を実施、
また音声データを削除。

## ５．フレームワーク事前調査
今回使用するMMDetectionフレームワークのチュートリアルを再現する。 
これによりフレームワークの使い方を身に付けるとともに、正常にインスタンス・セグメンテーションが実行
できる環境を構築する。 
再現実行するチュートリアルは、google Colab環境でGPUを使用することを前提とし、Mask R-CNNを
使ったものを選定する。 
https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb 
この実施結果を[”05prelim_survey.ipynb”](https://github.com/nob-fu/LABOLO_TOMATO-fine-tuning-exercise/blob/main/05prelim_survey.ipynb) にて示す。 

【実施結果】
MMDetection v2.24.0 ～v2.25.0（現時点の最新リリース版）では、上記チュートリアルはエラーとなる。

下記の通りこのエラーを解消した結果、pretraind modelを使ってのセグメンテーション動作の確認、および新たなdatasetを使い、風船の有無のみのクラスをセグメントするために、ファインチューニング学習させる方法の動作確認を行った。

症状： tools/train.pyを実行すると、”### AttributeError: ‘ConfigDict’ object has no attribute ‘device’”エラーが発生する。  
なお、参照しているチュートリアルの実行環境はMMDetection v2.21.0であることが示されている。  
このエラーについては、v2.24.0リリース後、たびたびＧｉｔＨｕｂ上issueとして報告され、対策提案がなされている。その対策を実施しようとしたが、自分の環境では解決しなかった。  
対策：MMDetection v2.23.0にバージョンダウンすることで解決し、チュートリアルの結果を確認できた。  
なおその際は、mmcvについても最新版（1.5.x）ではなく、1.3.17にバージョンダウンさせる必要がある。  

## ６．LaboloTomato事前学習モデルの検証
実施結果を[”06LaboroTomato_verif.ipynb”](https://github.com/nob-fu/LABOLO_TOMATO-fine-tuning-exercise/blob/main/06LaboloTomat_verif.ipynb)にて示す。  
1. MMDetectionのロード  
2. LaboroTomato model実行環境の再現  
3. LaboroTomato/test データによる正当性の確認（validation）  
画像161枚、検証時間：0.1～0.3task/sなので、画像1枚当たり3~10秒程度を要している。  
Average PrecisionはBoundary Boxについて64.7%、セグメンテーションについて66%。  
LaboroTomato GitHub上に記載された訓練時データでは「bbox AP:64.3, mask AP:65.7」なので、  
ほぼ再現されていると判断できる。 

|　　 |　　 |　　 |　　|
|---------------|--------:|---------------|--------:|
|'bbox_mAP' |0.647 |'segm_mAP’ |0.66|
|'bbox_mAP_50’ |0.822 |'segm_mAP_50' |0.818|
|'bbox_mAP_75' |0.735 |'segm_mAP_75' |0.736|
|'bbox_mAP_s' |0.0 |'segm_mAP_s' |0.0|
|'bbox_mAP_m' |0.146 |'segm_mAP_m' |0.131|
|'bbox_mAP_l' |0.681 |'segm_mAP_l' |0.697|

4. 新たに準備したデータによる検証（verification）

| File名      |結果     |記事                  | 
|:------------:|:---------:|:---------------------|
|eval_001.jpg| × |水滴がつくと正しく判定できない。水滴をトマトと誤検出。|
|eval_002.jpg| △ |奥の2個は葉っぱ、水滴に紛れて認識できていない|
|eval_003.jpg| △ |光線、枝の影などの影響か、2個重なっていると誤検出|
|eval_004.jpg| △ |茎やヘタ部分で仕切られると、複数有りと誤検出|
|eval_005.jpg| △ |同上|
|eval_006.jpg| 〇 |箱に数多く有るが良好に検出、黄色トマトは未成熟と認識|
|eval_007.jpg| 〇 |    |
|eval_008.jpg| × |ピンぼけ写真だとかなり未検出、またミニを通常サイズと誤認識している|
|eval_009.jpg| × |リンゴは学習していないのでトマトと誤認識される|
|eval_010.jpg| × |同上|
