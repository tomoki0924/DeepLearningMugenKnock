# ディープラーニング∞本ノック!!


ディープラーニング∞本（？）ノックぅぅ

まだ作成中なのであしからず

何問になるか分からないので∞本になってます。多分これからいろんな技術が出るからどんどん更新する予定でっす。
これはイモリと一緒にディープラーニングの基礎からDLのライブラリの扱い、どういうDLの論文があったかを実装しながら学んでいくための問題集です。本とか論文読んだだけじゃ机上の空想でしかないので、ネットワークの作成や学習率などのハイパーパラメータの設定を自分の手を動かしながら勉強するための問題集です。

**問題集として使ってもテンプレやチートシートとして使っても使い方は自由です！！！！**

僕が研究室で３年修行してディープラーニングで必要そうだなーと思ったものを集めてます。

例えば研究してて提案手法を急にKerasでやればとか簡単に言うけどそんなすぐにできるかいってよくあると思うんだけどそういうのにもすぐ対応できる力を身につけるためのものだとも思ってます。

- **内容はいろいろな文献を調べて載っけてるので正しくないものもあるかもしれないので注意して下さい。もし間違ってたらプルリク下さい笑**
- 【注意】このページを利用して、または関して生じた事に関しては、**私は一切責任を負いません。** すべて **自己責任** でお願い致します。
- コードの書き方は私の趣向がけっこう出てるので、この書き方キモってなったら自分の書き方でやっていってください。答えはあくまで参考です。FWによってチョクチョク実装に小さな違いがあるのでそこはご愛嬌
- なんとなく本とか買わずにDLを勉強したいーーーって人向けだと思う

## Related

画像処理ノックはこっち

> [画像処理100本ノック!!](https://github.com/yoyoyo-yo/Gasyori100knock)

## Recent
- 2019.7.17 [Pytorch] VAEを追加  [keras, tensorflow, chainer] CGAN(MNIST)を追加
- 2019.7.5 [pytorch, keras]ResNet18, 34, 101, 152を追加
- 2019.6.16 [pytorch, tensorflow, keras, chainer] ResNet50を追加
- 2019.6.9 [tensorflow] DCGANを追加
- 2019.6.7 [Pytorch, tensorflow, keras, chainer]GoogleNet-v1(Inception)を追加
- 2019.5.26 [tensorflow] DCGAN, Conditional GANを追加
- 2019.5.19 [Keras, Chainer] ConditionalGANを追加
- 2019.5.18 [データセット準備] MNIST, [Pytorch]ConditionalGANを追加
- 2019.5.2 [データセット準備] Cifar10、[AutoEncoder, ConvAutoEncoder, GAN, DCGAN]Cifar10を追加
- 2019.3.31 [画像認識モデル] APIを追加
- 2019.3.19 [Pytorch][Chainer] GAN, DCGANを追加
- 2019.3.17 Pooling layerを追加したけど、あとからクラス化と学習を追加する予定
- 2019.3.17 seq2seq, convolutional layer を追加
- 2019.3.16 ニューラルネットをクラス化　を追加
- 2019.3.13 パーセプトロン系を追加
- 2019.3.12 AutoEncoder, ConvAutoEncoder, パーセプトロンを追加
- 2019.3.9 GAN, DCGANを追加
- 2019.3.6 RNN, LSTM, BDLSTMを追加
- 2019.3.5 AutoEncoder, RNNを追加　
- 2019.3.4 データ拡張・回転を追加
- 2019.3.3 UNetを追加

## 環境設定

Python-3.6でやって下さい。(解答はPython-3.6で用意してます)

### 1. Minicondaのインストール

https://conda.io/miniconda.html のサイトからMinicondaをインストールします。これはWindowでもMacOSでも可能です。Minicondaがインストールできたら、端末(Windowでは端末、MacOSではターミナル)を開き、以下コマンドで仮想環境を作成します。**もしくはGoogle colabolatoryを使って見て下さい。GPUが使えます。**

```bash
$ conda create python=3.6 -n dlmugenknock
```

作成できたら、以下コマンドで仮想環境を動作します。

```bash
$ source activate dlmugenknock
```

するとこうなります。

```bash
(dlmugenknock) :~/work_space/DeepLearningMugenKnock/ :$ 
```

### 2. gitのインストール

gitをインストールします。そして、端末を開いて、以下のコマンドを実行します。このコマンドでこのディレクトリを丸ごと自分のパソコンにコピーできます。

```bash
$ git clone https://github.com/yoyoyo-yo/DeepLearningMugenKnock.git
```

### 3. パッケージのインストール

以下のコマンドで必要なパッケージをインストールします。これで準備は完了です！！


```bash
$ pip install -r requirements.txt
```

## フレームワーク早見表

- [**CNN・フレームワークの使い方編**](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_howto)
- [**共通事項**](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_howto)

| | PyTorch | Tensorflow | Keras | Chainer | Caffe |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 入力 | [mb,c,h,w] | [mb, h, w, c] | [mb, h, w, c] | [mc, c, h, w] | [mb, c, h, w] |
| 教師ラベル | index [mb] | onehot [mb, cls] | onehot [mb, cls] | index [mb] | index [mb] |
| 速度 | まあまあ早い | 早い | 早い | 普通 | まあまあ早い？ |
| how to | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_pytorch.md) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_tensorflow.md)  |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_keras.md) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_chainer.md) | [✓install(Docker)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_caffe_install_docker.md) <br>  [✓install(Native)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_caffe_install_native.md) |
| sample | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_pytorch.py)  | [✓(slim)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_tensorflow_slim.py) <br> [✓(layers)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_tensorflow_layers.py) <br> [✓(raw)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_tensorflow_raw.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_keras.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/main_chainer.py)  |

## 問題

詳細な問題内容は各ディレクトリのREADMEにあります。（ディレクトリで下にスクロールすればあります）

## 自分で全部実装編

### [理論編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory)

| 番号 | 問題 | 答え | | 番号 | 問題 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [パーセプトロン AND](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-and) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_1.py)
| 2 | [パーセプトロン 学習](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_2.py)
| 3 | [パーセプトロン 収束性](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%8F%8E%E6%9D%9F%E6%80%A7) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_3.py)
| 4 | [パーセプトロン Sigmoid](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-sigmoid) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_sigmoid.py)
| 5 | [パーセプトロン バイアス](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-bias) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_sigmoid_bias.py)
| 6 | [パーセプトロン OR](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-or) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_or.py)
| 7 | [パーセプトロン NOT](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-not) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_not.py)
| 8 | [パーセプトロン XOR](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-xor%E3%82%B2%E3%83%BC%E3%83%88) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/perceptron_xor.py )
| 9 | [多層パーセプトロン FeedForward](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-feedforward) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/multi_perceptron_1.py)
| 10 | [多層パーセプトロン 学習](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/multi_perceptron_2.py)
| 11 | [更に多層パーセプトロン](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E6%9B%B4%E3%81%AB%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/multi_perceptron_3.py)
| 12 | [ニューラルネットのクラス化](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory#q-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%81%AE%E3%82%AF%E3%83%A9%E3%82%B9%E5%8C%96) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory/answers/multi_perceptron_class.py)

### [理論編2](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/)

| 番号 | 問題 | 答え | | 番号 | 問題 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 | [画像認識](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#%E7%94%BB%E5%83%8F%E8%AA%8D%E8%AD%98) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/neuralnet_classification.py)
| 14 | [誤差関数](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#%E8%AA%A4%E5%B7%AE%E9%96%A2%E6%95%B0) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/neuralnet_loss.py)
|  | [Sigmoid Cross Entropy](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#sigmoid-cross-entropy) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/neuralnet_sce.py)
|  | [Convolutional Layer](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#convolutional-layer) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/conv_kernel.py)
|  | [Padding](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#padding) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/conv_pad.py)
|  | [Stride](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2/#stride) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/conv_stride.py)
|  | [Max-pooling layer](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2#max-pooling-layer) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/maxpool.py)
|  | [Average-pooling layer](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_theory2#average-pooling-layer) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_theory2/answers/avepool.py)


### [ディープラーニングをやる前の準備編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare)

|番号|問題| 答え | | 番号|問題|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [データセットの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-1-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_data_load.py)
| 2 | [ミニバッチの作成](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-2-%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E4%BD%9C%E6%88%90) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_minibatch.py)
| 3 | [イテレーション・エポック](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-3-%E3%82%A4%E3%83%86%E3%83%AC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%A8%E3%82%A8%E3%83%9D%E3%83%83%E3%82%AF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_epoch.py)
| 4 | [データ拡張・水平反転](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q4-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%B7%A6%E5%8F%B3%E5%8F%8D%E8%BB%A2) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_hf.py)
| 5 | [データ拡張・上下反転](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q5-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E4%B8%8A%E4%B8%8B%E5%8F%8D%E8%BB%A2) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_vf.py)
| 6 | [データ拡張・回転](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q6-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%9B%9E%E8%BB%A2) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/answer_rotation.py)
|  | [MNIST Step.1 ダウンロード](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-mnist-10-step1-%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_mnist_step1.py)
|  | [MNIST Step.2 学習データの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-mnist-10-step2-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_mnist_step2.py)
|  | [MNIST Step.Final テストデータの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-mnist-10-stepfinal-%E3%83%86%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_mnist.py)
|  | [CIFAR-10 Step.1 ダウンロード](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-cifar-10-step1-%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_cifar10_step1.py)
|  | [CIFAR-10 Step.2 学習データの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-cifar-10-step2-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_cifar10_step2.py)
|  | [CIFAR-10 Step.Final テストデータの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q-cifar-10-stepfinal-%E3%83%86%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answers/load_cifar10.py)


## 自分でネットワーク組む編

ここから基本的に、「python answers/##.py --train --test」と打てば動きます。

### [画像認識編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model)

| 問題 |  PyTorch | TensorFlow | Keras | Chainer |
|:---:|:---:|:---:|:---:|:---:|
| [ API ](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#api) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/api_pytorch.py) | 
| [LeNet](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-lenet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/lenet_chainer.py) | 
| [AlexNet](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-alexnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/alexnet_chainer.py) | 
| [ZFNet](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-zfnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/zfnet_chainer.py) | 
| [Global Average Pooling](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-zfnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/gap_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/gap_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/gap_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/gap_chainer.py) | 
| [Network in network](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-network-in-network) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/nin_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/nin_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/nin_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/nin_chainer.py) | 
| [VGG16](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-vgg16) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg16_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg16_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg16_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg16_chainer.py) | 
| [VGG19](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-vgg19) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg19_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg19_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg19_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/vgg19_chainer.py) | 
| [モデルの書き方の簡潔化](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-vgg19) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/easy_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/easy_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/easy_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/easy_chainer.py) | 
| [GoogLeNet-v1(Inception)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-googlenet-v1) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/googletnetv1_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/googlenetv1_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/googlenetv1_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/googlenetv1_chainer.py) | 
| [Batch Normalization](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-vgg19) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/bn_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/bn_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/bn_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/bn_chainer.py) |
| [ResNet-18](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-resnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res18_pytorch.py) | | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res18_keras.py) | |
| [ResNet-34](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-resnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res34_pytorch.py) | | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res34_keras.py) | | 
| [ResNet-50](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-resnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res50_pytorch.py) | [✓(not good)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res50_tensorflow_layers.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res50_keras.py) | [✓(not good)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res50_chainer.py) | 
| [ResNet-101](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-resnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res101_pytorch.py) | | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res101_keras.py) | |
| [ResNet-152](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_model#q-resnet) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res152_pytorch.py) | | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_model/answers/res152_keras.py) | | 


### [Semantic Segmentation編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg)
| 問題 |  PyTorch | TensorFlow | Keras | Chainer |
|:---:|:---:|:---:|:---:|:---:|
| [SemanticSegmentationとは?](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#semanticsegmentation%E3%81%A8%E3%81%AF) |
| [Binalization Step.1. データセット読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#binalization%E3%81%AB%E3%82%88%E3%82%8Bsemasegstep1-%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_dataset_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_dataset_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_dataset_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_dataset_chainer.py) |
| [Binalization Step.2. 学習時のLoss計算](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#binalization-step2-%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%AEloss%E8%A8%88%E7%AE%97)| [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_loss_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_loss_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_loss_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_loss_chainer.py) | 
| [Binalization Step.3. テスト時の予測結果の表示](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#binalization%E3%81%AB%E3%82%88%E3%82%8Bsemasegstep3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_test_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_test_tensorflow_slim.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_test_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/bin_test_chainer.py) |  
| [SemanticSegmentation Step.1. データセット読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#semantic-segmentation-step1-%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_dataset_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_dataset_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_dataset_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_dataset_chainer.py) |
| [SemanticSegmentation Step.2. 学習時のLoss計算](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#semantic-segmentation-step2-%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%AEloss%E8%A8%88%E7%AE%97)| [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_loss_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_loss_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_loss_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_loss_chainer.py) |
| [SemanticSegmentation Step.3. テスト時の予測結果の表示](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#semantic-segmentation-step3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_test_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_test_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_test_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/semaseg_test_chainer.py) |
| [UpSampling手法1. NearestNeighbor補間](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#upsampling%E6%89%8B%E6%B3%951-nearestneighbor%E8%A3%9C%E9%96%93) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/nearest_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/nearest_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/nearest_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/nearest_chainer.py) |
| [UpSampling手法2. Transposed convolution](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#upsampling%E6%89%8B%E6%B3%952-transposed-convolution) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/transposeconv_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/transposeconv_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/transposeconv_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/transposeconv_chainer.py) |
| [特徴マップのconcat](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#%E7%89%B9%E5%BE%B4%E3%83%9E%E3%83%83%E3%83%97%E3%81%AEconcat) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/concat_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/concat_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/concat_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/concat_chainer.py) |
| [UNet](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#unet) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unet_pytorch.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unet_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unet_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unet_chainer.py) |
| [UNet風モデル](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_semaseg#unet%E9%A2%A8%E3%83%A2%E3%83%87%E3%83%AB)|  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unetlike_pytorch.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unetlike_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unetlike_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/unetlike_chainer.py) |


### [画像生成編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate)
| 問題 |  PyTorch | TensorFlow | Keras | Chainer |
|:---:|:---:|:---:|:---:|:---:|
| [AutoEncoder](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-auto-encoder) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/ae_pytorch.py) | [✓?](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_tensorflow_slim.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_chainer.py) |
| [AutoEncoder cifar10](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-auto-encoder) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/ae_cifar10_pytorch.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_cifar10_tensorflow_slim.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_cifar10_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/ae_cifar10_chainer.py) |
| [ConvolutionalAutoEncoder](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-convolutional-auto-encoder) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/convae_pytorch.py) |  [✓?](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_chainer.py) |
| [ConvolutionalAutoEncoder cifar10](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-convolutional-auto-encoder) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/convae_cifar10_pytorch.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_cifar10_tensorflow_slim.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_cifar10_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/convae_cifar10_chainer.py) |
| [VAE (Variational AutoEncoder) MNIST](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-auto-encoder) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_semaseg/answers/vae_mnist_pytorch.py) |
| [GAN](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-gan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_pytorch.py) |  [✓ not good](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_chainer.py) |
| [GAN cifar10](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-gan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_cifar10_pytorch.py) | [✓ failed](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_cifar10_tensorflow_slim.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_cifar10_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/gan_cifar10_chainer.py) |
| [DCGAN](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-dcgan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_chainer.py)
| [DCGAN cifar10](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-dcgan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_cifar10_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_cifar10_tensorflow_slim.py)  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_cifar10_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/dcgan_cifar10_chainer.py) |
| [Conditional GAN mnist](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#q-conditional-gan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_mnist_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_mnist_tensorflow_slim.py)   | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_mnist_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_mnist_chainer.py) |
| [Conditional GAN cifar10](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_imageGenerate#conditional-gan) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_cifar10_pytorch.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_cifar10_tensorflow_slim.py)   | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_cifar10_keras.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_imageGenerate/answers/cgan_cifar10_chainer.py) |

### [画像処理編]()

### [言語処理編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp)
| 問題 |  PyTorch | TensorFlow | Keras | Chainer |
|:---:|:---:|:---:|:---:|:---:|
| [1 hotベクトル化](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-1hot%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E5%8C%96) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/onehot.py) 
| [RNN (Many-to-one) Step.1. 学習 ](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-rnn-many-to-one-step1-%E5%AD%A6%E7%BF%92) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_tensorflow_slim.py) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_keras.py) | |
| [RNN (Many-to-one) Step.2. テスト](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-rnn-many-to-one-step2-%E3%83%86%E3%82%B9%E3%83%88) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_keras.py) |  |
| [LSTM (Many-to-one)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-lstm-many-to-one) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_chainer.py) |
| [Bi-directional LSTM (Many-to-one)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-bi-directional-lstm-many-to-one) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/bdlstm_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/bdlstm_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/bdlstm_keras.py) | [✓?](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/bdlstm_chainer.py) |
| [GRU (Many-to-one)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-gru-many-to-one) |  [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/gru_pytorch.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/gru_tensorflow_slim.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/gru_keras.py) | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/gru_chainer.py) |
| [Seq2seq (Many-to-many)](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_nlp#q-seq2seq-many-to-many) |  |  | [✓](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/seq2seq_keras.py) |

## Citation

```bash
@article{yoyoyo-yoDeepLearningMugenKnock,
    Author = {yoyoyo-yo},
    Title = {DeepLearningMugenKnock},
    Journal = {https://github.com/yoyoyo-yo/DeepLearningMugenKnock},
    Year = {2019}
}
```

## License

&copy; Yoshito Nagaoka All Rights Reserved.

This is under MIT License.

https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/LICENSE
