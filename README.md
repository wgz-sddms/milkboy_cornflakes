## コーンフレーク分類器（ミルクボーイ）
オカンが言ってた特徴を入力すると，それがコーンフレークなのか否かツッコミを入れてくれる.

### 動作テスト

```
python milkboy_try.py
```
上記コマンドを実行し，特徴を角刈りに教えてあげる．

(注) [学習済みword2vec](https://github.com/singletongue/WikiEntVec)
を用いているため，レポジトリに準備する必要がある．（ソースコード参照)

### 各スクリプト説明
- `milkboy_data.csv`
  - データファイル．データ数は100件，実際の漫才の書き起こしに加え，wikipedia等から寄せ集めたもの（個人的見解含）

- `milkboy_data.py`
  - データ前処理やデータセット作成の関数
  
- `milkboy_train.py`
  - 分類器の訓練と検証．学習済みword2vecを用いて簡易なニューラルネットワークで学習させている．
  
- `entity_vector（フォルダ）`
  - アップロードしていないが，このフォルダに[学習済みword2vec](https://github.com/singletongue/WikiEntVec)を置いてある．
