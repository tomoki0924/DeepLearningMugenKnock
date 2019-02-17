# Q. 有名モデル実装編

ここではCNNの有名モデルを自分の手で実装していく。フレームワークは自由だが、**とりあえずPyTorch, Tensorflow, Keras, Chainer全部で実装せよ。**
ネットワークを作ったら、学習率やイテレーションを変えて、テストデータセット *../Dataset/test/images* でテストせよ。

## Q. LeNet

論文 >> http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

これが原初のモデル。MNISTと呼ばれる0から9までの手書き数字の判別で使われたCNNモデル。これを実装せよ。LeNetはMNIST用に入力サイズが32x32となっているが、ここで用意しているデータセットは128x128サイズである。**よって学習時のデータの入力サイズを32x32にリサイズする必要がある。**

構造は以下の様になる。

| Layer | カーネルサイズ | フィルタ数 |  ストライド| パディング |  活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 32 x 32 x 3(入力サイズ) |
| Convolution | 5 x 5 |  6 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| Convolution | 5 x 5 | 16 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| MultiLayerPerceptron | 120 | - | - | - | - | - |
| MultiLayerPerceptron |  64 | - | - | - | - | - |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|



答え
- Pytorch [answers/lenet_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_pytorch.py)
- Tensorflow [answers/lenet_tensorflow_layers.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_tensorflow_layers.py)
- Keras [answers/lenet_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_keras.py)
- chainer [answers/lenet_chainer.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_chainer.py)

## Q. AlexNet

論文 >> https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

ディープラーニングを流行らせた張本人モデル。ImageNetという画像認識のコンペILSVRC2012で圧倒的一位で優勝したことから現在のディープラーニングブームが起こった。これを実装せよ。
AlexNetでは*Local Response Normalization* という特別な正規化Layerがある。
pytorchでは *torch.nn.modules.normalization.LocalResponseNorm()*、chainerでは*chainer.functions.local_response_normalization()* Tensorflowでは*tf.nn.local_response_normalization()* で実装がある。Kerasにはないので実装しなくてもよい。
LRNは効果が薄いことから最近ではほとんど使われない。こういうのもあったんだなあ程度に覚えておくといいと思う。

ただし学習データの枚数が少ないので学習が進まないので、精度を上げたいときは自分で学習データを増やすか、パラメータ数を変えるなどの工夫が必要なので注意。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 227 x 227 x 3 (入力サイズ) |
| Convolution | 11 x 11 | 96 | 4 | 0 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 5 x 5 | 256 | 1 | 1 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|


答え
- Pytorch [answers/alexnet_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_pytorch.py)
- Tensorflow [answers/alexnet_tensorflow_layers.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_tensorflow_layers.py)
- Keras [answers/alexnet_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_keras.py)
- chainer [answers/alexnet_chainer.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_chainer.py)


## Q. ZFNet

論文 >> https://arxiv.org/abs/1311.2901

ILSVRC2013で一位をとったモデル。AlexNetと構造が似ている。
Alexnetの最初のconvlutionを7x7のカーネルにして、ストライドを2に変更している。そのかわりに２つ目のconvolutionのストライドを2にしている。こうすることで、大きなカーネルサイズによる画像の周波数取得を変えている。論文ではCNNが画像認識を行うまでの解析を主張している。

ここらへんから計算時間がめちゃくちゃ増えるので、GPU使用をおすすめ。もしくは入力サイズを112x112とか小さくすることを推奨。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 224 x 224 x 3 (入力サイズ) |
| Convolution | 7 x 7 | 96 | 2 | 0 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 5 x 5 | 256 | 2 | 1 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|

答え
- Pytorch [answers/alexnet_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_pytorch.py)
- Tensorflow [answers/alexnet_tensorflow_layers.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_tensorflow_layers.py)
- Keras [answers/alexnet_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_keras.py)
- chainer [answers/alexnet_chainer.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_chainer.py)

