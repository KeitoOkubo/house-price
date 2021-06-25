# ライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# 解析を行うデータの読み込み
df = pd.read_csv("./train.csv")

# XにOverallQual、yにSalePriceをセットする
# すなわちOverallQualとSalePriceの関係について調べる
X = df[["OverallQual"]].values
y = df["SalePrice"].values

# アルゴリズムに線形回帰(Linear Regression)を採用
# Xとyに関して直線のモデルで関係を調べる
slr = LinearRegression()

# fit関数でモデル作成
slr.fit(X,y)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_[0]))

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))

# 散布図を描画
plt.scatter(X,y)
# 折れ線グラフを描画
plt.plot(X,slr.predict(X),color='red')
# 2つのグラフを重ねて表示
plt.show()



# テストデータの読込
df_test = pd.read_csv("./test.csv")
# テストデータの OverallQual の値をセット
X_test = df_test[["OverallQual"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット
df_test["SalePrice"] = y_test_pred



# Id, SalePriceの2列だけ表示
df_test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
