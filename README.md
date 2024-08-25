コーヒーマシンの牛乳消費量予測
このリポジトリは、コーヒーマシンの売上データを基に、日々の牛乳消費量を予測する機械学習プロジェクトを含んでいます。予測モデルは、Pythonとscikit-learnを使用して構築されています。

プロジェクト概要
コンビニでは、毎日コーヒーマシンの清掃のため、牛乳の入れ替えの最大量の牛乳が廃棄されています。
資源を効率的に管理することが重要です。その中でも牛乳は多くのコーヒー飲料の主要な材料です。このプロジェクトでは、コーヒーマシンの牛乳消費量を予測し、在庫を最適化し、無駄を減らすことを目指しています。

主な機能
予測モデル: scikit-learnを利用して、日々の牛乳消費量を予測する回帰モデルを構築。
データ処理: 実際に使用された売上データは機密保持のため共有できませんが、類似のデータセットで動作するようにプロジェクトが構成されています。
可視化: モデルの予測結果を示すグラフを、figディレクトリ内に含んでいます。
リポジトリ構成


├── prediction_test.py     #予測モデルを含むメインスクリプト
├── README.md              #プロジェクトのドキュメント
├── 解説用パワーポイント.pptx #プロジェクトのドキュメント(背景などを詳しく記載したもの)
└── fig/                   #出力グラフを含むディレクトリ
    ├── Permutation_Importance.png
    ├── heatmap.png #各説明変数と目的変数の相関係数のヒートマップ
    └── output.png

prediction_test.py
このスクリプトには、牛乳消費量予測モデルの主要なロジックが含まれています。内容としては以下を含みます:

データの前処理
scikit-learnを使用したモデルのトレーニング
予測生成
予測結果のグラフ可視化
fig/
このディレクトリには、モデルによって生成された出力グラフが含まれています。
Permutation_Importance.png 特徴量重要度の可視化
heatmap.png 　　　　　　　　　各説明変数と目的変数の相関係数のヒートマップ
output.png　　　　　　　　　  機械学習モデルの予測結果を出力したグラフ

実行方法
リポジトリをクローンする:
git clone https://github.com/yourusername/repository-name.git
プロジェクトディレクトリに移動:
cd repository-name
メインスクリプトを実行:
python prediction_test.py

依存関係
Python 3.9.19
以下に自分のconda listを記載しました。必要に応じてライブラリをインストールしてください！
# Name                    Version                   Build  Channel
alembic                   1.7.5              pyhd3eb1b0_1  
anyio                     3.6.1                    pypi_0    pypi
appnope                   0.1.3                    pypi_0    pypi
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0           py39hca72f7f_0  
asttokens                 2.0.5              pyhd3eb1b0_0  
attrs                     21.4.0             pyhd3eb1b0_0  
autopage                  0.5.1              pyhd8ed1ab_0    conda-forge
babel                     2.10.3                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.11.1           py39hecd8cb5_0  
blas                      1.0                         mkl  
bleach                    5.0.1                    pypi_0    pypi
bottleneck                1.3.5            py39h67323c0_0  
brotli                    1.0.9                hca72f7f_7  
brotli-bin                1.0.9                hca72f7f_7  
bzip2                     1.0.8                h1de35cc_0  
c-ares                    1.19.1               h6c40b1e_0  
ca-certificates           2024.7.2             hecd8cb5_0  
cairo                     1.16.0               h3ce6f7e_5  
cartopy                   0.20.3                   pypi_0    pypi
certifi                   2024.7.4         py39hecd8cb5_0  
cffi                      1.15.1           py39hc55c11b_0  
cftime                    1.5.1.1          py39h67323c0_0  
charset-normalizer        2.1.0                    pypi_0    pypi
click                     8.1.3                    pypi_0    pypi
cliff                     4.0.0              pyhd8ed1ab_0    conda-forge
cmaes                     0.8.2              pyh44b312d_0    conda-forge
cmd2                      2.4.2            py39h6e9494a_0    conda-forge
colorlog                  5.0.1            py39hecd8cb5_1  
curl                      8.7.1                h04015c4_0  
cycler                    0.11.0             pyhd3eb1b0_0  
debugpy                   1.6.0                    pypi_0    pypi
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
eli5                      0.13.0             pyhd8ed1ab_0    conda-forge
entrypoints               0.4              py39hecd8cb5_0  
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.6.2                hcec6c5f_0  
fastjsonschema            2.15.3                   pypi_0    pypi
fftw                      3.3.9                h9ed2024_1  
flask                     2.1.2                    pypi_0    pypi
font-ttf-dejavu-sans-mono 2.37                 hd3eb1b0_0  
font-ttf-inconsolata      2.001                hcb22688_0  
font-ttf-source-code-pro  2.030                hd3eb1b0_0  
font-ttf-ubuntu           0.83                 h8b1ccd4_0  
fontconfig                2.14.1               hedf32ac_1  
fonts-anaconda            1                    h8fa9717_0  
fonts-conda-ecosystem     1                    hd3eb1b0_0  
fonttools                 4.33.3                   pypi_0    pypi
freetype                  2.11.0               hd8bbffd_0  
fribidi                   1.0.10               haf1e3a3_0  
gdk-pixbuf                2.42.10              h46256e1_1  
geos                      3.9.1                h23ab428_0  
gettext                   0.21.0               h7535e17_0  
giflib                    5.2.1                haf1e3a3_0  
glib                      2.78.4               hcec6c5f_0  
glib-tools                2.78.4               hcec6c5f_0  
graphite2                 1.3.14               he9d5cce_1  
graphviz                  2.50.0               h196fa6a_0  
greenlet                  1.1.1            py39h23ab428_0  
gts                       0.7.6                h6759243_3  
harfbuzz                  4.3.0                hffc734d_1  
hdf4                      4.2.13               h39711bb_2  
hdf5                      1.10.6               h10fe05b_1  
icu                       58.2                 h0a44026_3  
idna                      3.3                      pypi_0    pypi
importlib-metadata        4.12.0                   pypi_0    pypi
importlib_metadata        4.11.3               hd3eb1b0_0  
importlib_resources       5.2.0              pyhd3eb1b0_1  
intel-openmp              2021.4.0          hecd8cb5_3538  
ipykernel                 6.15.0                   pypi_0    pypi
ipython                   8.4.0            py39hecd8cb5_0  
ipython_genutils          0.2.0              pyhd3eb1b0_1  
itsdangerous              2.1.2                    pypi_0    pypi
jedi                      0.18.1           py39hecd8cb5_1  
jinja2                    3.1.2                    pypi_0    pypi
joblib                    1.1.0              pyhd3eb1b0_0  
jpeg                      9e                   hca72f7f_0  
json5                     0.9.8                    pypi_0    pypi
jsonschema                4.6.1                    pypi_0    pypi
jupyter-client            7.3.4                    pypi_0    pypi
jupyter-core              4.10.0                   pypi_0    pypi
jupyter-server            1.18.0                   pypi_0    pypi
jupyter_client            7.3.5            py39hecd8cb5_0  
jupyter_core              4.11.1           py39hecd8cb5_0  
jupyterlab                3.4.3                    pypi_0    pypi
jupyterlab-pygments       0.2.2                    pypi_0    pypi
jupyterlab-server         2.14.0                   pypi_0    pypi
jupyterlab_pygments       0.1.2                      py_0  
kiwisolver                1.4.3                    pypi_0    pypi
krb5                      1.20.1               h049b76e_0    conda-forge
lcms2                     2.12                 hf1fd2bf_0  
lerc                      3.0                  he9d5cce_0  
libbrotlicommon           1.0.9                hca72f7f_7  
libbrotlidec              1.0.9                hca72f7f_7  
libbrotlienc              1.0.9                hca72f7f_7  
libcurl                   8.7.1                hf20ceda_0  
libcxx                    14.0.6               h9765a3e_0  
libdeflate                1.8                  h9ed2024_5  
libedit                   3.1.20210910         hca72f7f_0  
libev                     4.33                 h9ed2024_1  
libffi                    3.4.4                hecd8cb5_1  
libgd                     2.3.3                hcec6c5f_3  
libgfortran               5.0.0           11_3_0_hecd8cb5_28  
libgfortran5              11.3.0              h9dfd629_28  
libglib                   2.78.4               h19e1a8f_0  
libiconv                  1.16                 h6c40b1e_3  
libnetcdf                 4.8.1                h24cb85c_1  
libnghttp2                1.57.0               h9beae6a_0  
libpng                    1.6.39               h6c40b1e_0  
librsvg                   2.54.4               h52d90eb_0  
libsodium                 1.0.18               h1de35cc_0  
libssh2                   1.11.0               hf20ceda_0  
libtiff                   4.4.0                h2ef1027_0  
libtool                   2.4.6             hcec6c5f_1009  
libwebp                   1.2.4                h56c3ce4_0  
libwebp-base              1.2.4                hca72f7f_0  
libxml2                   2.9.14               hbf8cd5e_0  
libzip                    1.8.0                h29ab7a1_1  
lightgbm                  3.2.1            py39h23ab428_0  
llvm-openmp               14.0.6               h0dcd299_0  
lxml                      4.9.1                    pypi_0    pypi
lz4-c                     1.9.3                h23ab428_1  
mako                      1.2.3            py39hecd8cb5_0  
markupsafe                2.1.1            py39hca72f7f_0  
matplotlib                3.5.2            py39hecd8cb5_0  
matplotlib-base           3.5.2            py39hfb0c5b7_0  
matplotlib-inline         0.1.3                    pypi_0    pypi
mistune                   0.8.4           py39h9ed2024_1000  
mkl                       2021.4.0           hecd8cb5_637  
mkl-service               2.4.0            py39h9ed2024_0  
mkl_fft                   1.3.1            py39h4ab4a9b_0  
mkl_random                1.2.2            py39hb2f4e1b_0  
munkres                   1.1.4                      py_0  
nbclassic                 0.4.0                    pypi_0    pypi
nbclient                  0.6.6                    pypi_0    pypi
nbconvert                 6.5.0                    pypi_0    pypi
nbformat                  5.4.0                    pypi_0    pypi
ncurses                   6.4                  hcec6c5f_0  
nest-asyncio              1.5.5            py39hecd8cb5_0  
netcdf4                   1.5.7            py39h4a1dd59_1  
notebook                  6.4.12           py39hecd8cb5_0  
notebook-shim             0.1.0                    pypi_0    pypi
numexpr                   2.8.3            py39h2e5f0a9_0  
numpy                     1.21.5           py39h2e5f0a9_3  
numpy-base                1.21.5           py39h3b1a694_3  
openssl                   3.0.14               h46256e1_0  
optuna                    3.0.2              pyhd8ed1ab_0    conda-forge
packaging                 21.3               pyhd3eb1b0_0  
pandas                    1.4.3                    pypi_0    pypi
pandocfilters             1.5.0              pyhd3eb1b0_0  
pango                     1.50.7               h80fe9ab_0  
parso                     0.8.3              pyhd3eb1b0_0  
patsy                     0.5.2            py39hecd8cb5_1    anaconda
pbr                       5.6.0              pyhd3eb1b0_0  
pcre2                     10.42                h9b97e30_1  
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    9.2.0                    pypi_0    pypi
pip                       22.2.2           py39hecd8cb5_0  
pixman                    0.40.0               h9ed2024_1  
prettytable               3.4.1              pyhd8ed1ab_0    conda-forge
proj                      8.2.1                hd69def0_0  
prometheus_client         0.14.1           py39hecd8cb5_0  
prompt-toolkit            3.0.30                   pypi_0    pypi
psutil                    5.9.1                    pypi_0    pypi
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.12.0                   pypi_0    pypi
pyparsing                 3.0.9            py39hecd8cb5_0  
pyperclip                 1.8.2              pyhd8ed1ab_2    conda-forge
pyproj                    3.3.1                    pypi_0    pypi
pyrsistent                0.18.1                   pypi_0    pypi
pyshp                     2.3.0                    pypi_0    pypi
python                    3.9.19               h5ee71fb_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-fastjsonschema     2.16.2           py39hecd8cb5_0  
python-graphviz           0.20.1           py39hecd8cb5_0  
python_abi                3.9                      2_cp39    conda-forge
pytz                      2022.1           py39hecd8cb5_0  
pyyaml                    6.0              py39hca72f7f_1  
pyzmq                     23.2.0           py39he9d5cce_0  
readline                  8.1.2                hca72f7f_1  
requests                  2.28.1                   pypi_0    pypi
scikit-learn              1.1.2            py39he9d5cce_0  
scipy                     1.7.3            py39h214d14d_2  
seaborn                   0.11.2             pyhd3eb1b0_0  
send2trash                1.8.0              pyhd3eb1b0_1  
setuptools                63.4.1           py39hecd8cb5_0  
shapely                   1.8.2                    pypi_0    pypi
singledispatch            3.7.0           pyhd3eb1b0_1001  
six                       1.16.0             pyhd3eb1b0_1  
sniffio                   1.2.0                    pypi_0    pypi
soupsieve                 2.3.2.post1              pypi_0    pypi
sqlalchemy                1.4.39           py39hca72f7f_0  
sqlite                    3.45.3               h6c40b1e_0  
stack-data                0.3.0                    pypi_0    pypi
stack_data                0.2.0              pyhd3eb1b0_0  
statsmodels               0.13.2           py39hca72f7f_0    anaconda
stevedore                 4.0.0              pyhd8ed1ab_0    conda-forge
tabulate                  0.9.0            py39hecd8cb5_0  
terminado                 0.15.0                   pypi_0    pypi
testpath                  0.6.0            py39hecd8cb5_0  
threadpoolctl             2.2.0              pyh0d69192_0  
tinycss2                  1.1.1                    pypi_0    pypi
tk                        8.6.12               h5d9f67b_0  
tornado                   6.2              py39hca72f7f_0  
tqdm                      4.64.1           py39hecd8cb5_0  
traitlets                 5.3.0                    pypi_0    pypi
typing-extensions         4.3.0            py39hecd8cb5_0  
typing_extensions         4.3.0            py39hecd8cb5_0  
tzdata                    2022c                h04d1e81_0  
urllib3                   1.26.9                   pypi_0    pypi
wcwidth                   0.2.5              pyhd3eb1b0_0  
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.3.3                    pypi_0    pypi
werkzeug                  2.1.2                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
xarray                    2022.3.0                 pypi_0    pypi
xz                        5.4.6                h6c40b1e_1  
yaml                      0.2.5                haf1e3a3_0  
zeromq                    4.3.4                h23ab428_0  
zipp                      3.8.0            py39hecd8cb5_0  
zlib                      1.2.13               h4b97444_1  
zstd                      1.5.2                hcb37349_0 

データプライバシーに関する注意
機密保持契約により、モデルのトレーニングやテストに使用した売上データをこのリポジトリに含めることはできません。しかし、コードはあなた自身のデータセットに簡単に適応できるように構成されています。main_program.py内のデータ読み込み部分をあなたのデータに置き換えてご使用ください。

将来的な改善点
パワーポイントに記載しています！そちらをご覧ださい！
