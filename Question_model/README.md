# Q. 有名モデル実装編

ここではCNNの有名モデルを自分の手で実装していく。フレームワークは自由だが、**とりあえずPyTorch, Tensorflow, Keras, Chainer全部で実装せよ。**

## Q.1. LeNet

これが原初のモデル。MNISTと呼ばれる0から9までの手書き数字の判別で使われたCNNモデル。これを実装せよ。LeNetはMNIST用に入力サイズが32x32となっているが、ここで用意しているデータセットは128x128サイズである。**よって学習時のデータの入力サイズを32x32にリサイズする必要がある。**

学習率やイテレーションを変えて、テストデータセット *../Dataset/test/images* ６枚中5枚以上判別できるようにせよ。 

構造は以下の様になる。

| | Layer | 入力サイズ | カーネルサイズ | 出力サイズ | ストライド| パディング | フィルタ数 | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Input | 32 x 32 x 3 |
| 1 | Convolution | 32 x 32 x 3 | 28 x 28 x 6 | 5 x 5 | 1 | 0 | 6 | - |
| 2 | MaxPooling | 28 x 28 x 6 | 14 x 14 x 6 | 2 x 2 | 2 | 0 | - | sigmoid |
| 3 | Convolution | 14 x 14 x 6 | 10 x 10 x 16 | 5 x 5 | 1 | 0 | 16 | - |
| 4 | MaxPooling | 10 x 10 x 16 | 5 x 5 x 16 | 2 x 2 | 2 | 0 | - | sigmoid |
| 5 | MultiLayerPerceptron | 500(5x5x16) | 120 | - | - | - | - | - |
| 6 | MultiLayerPerceptron | 120 | 64 | - | - | - | - | - |
| 7 | MultiLayerPerceptron | 64 | 2 (クラス) | - | - | - | - | Softmax|

元論文
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

答え
- Pytorch [answers/lenet_pytorch.py]()
- Tensorflow [answers/lenet_tensorflow.py]()
- Keras [answers/lenet_keras.py]()
- chainer [answers/lenet_chainer.py]()

