# CT_Lung_Segmentation
----------------------
* Kaggle CT Lung Segmentation 데이터 셋을 이용하여  
CT 폐 촬영 영상을 이용하여 실제 폐 영역만 추출하는 소스코드입니다.  
CNN을 사용하였으며, model.py 에서 모델의 구조를 확인하실 수 있습니다.  
모델의 구조는 기본적으로 CNN 기반의 인코더 - 디코더 구조를 가지고 있습니다.  
인코더 계층에서 총 4번의 Conv 레이어와 MaxPooling 을 거치며 피쳐를 추출하게 되고  
최종적으로 16*16 크기의 256개 피쳐맵이 생성됩니다.  
추출된 피쳐맵을 다시 디코더 레이어에 집어넣어 네번의 업샘플링, Conv 레이어를 거쳐  
폐 영역을 Segmentation 합니다.  

입력 이미지는 (256,256) 해상도이며 4번의 풀링 레이어를 통해 16 * 16 해상도의 256개 피쳐맵으로 변환되고  
이를 다시 디코더에서 256,256 크기의 단일 이미지로 복구하며 세그멘테이션이 이루어집니다.

# Test Result
-------------
![result_1.png](https://github.com/elensar92/CT_Lung_Segmentation/blob/master/Result/result_1.png?raw=true)
![result_26.png](https://github.com/elensar92/CT_Lung_Segmentation/blob/master/Result/result_26.png?raw=true)
![result_14.png](https://github.com/elensar92/CT_Lung_Segmentation/blob/master/Result/result_14.png?raw=true)

# Training
------------
* 위 신경망은 다음 *[데이터셋](https://www.kaggle.com/kmader/finding-lungs-in-ct-data) 을 이용하여 훈련하였습니다.  
합쳐서 300개 남짓한 CT 촬영 영상이지만 생각보다 굉장히 Segmentation이 잘 되는 편입니다.  
학습은 약 300개의 전체 데이터 중 10%를 Validation Data로 따로 뽑아 둔 뒤, 각각을 numpy 를 이용하여 npy 형태로 따로 저장하여  
학습 시에 사용하였습니다.  
전체 데이터에 대해 150번 학습을 수행하였으며, 배치사이즈는 데이터의 크기가 작아 다양하게 테스트 해보았으나 여기에 올린 것은 64 배치사이즈로 학습을
진행하였습니다.

# Enviroments 
--------------
* Ubuntu 18.04(Training)  
* Mac Catalina(10.15.1)(test) - MacBook Pro (Retina 13-inch, Mid 2014)  
윈도우에서는 테스트해보지 않았습니다.

# Dependencies
---------------
* Tensorflow 2.0
* Scikit Packages(learn,image etc...)
* Numpy
* Python 3 
* glob
* Matplotlib

