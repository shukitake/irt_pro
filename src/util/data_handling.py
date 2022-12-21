import os
import pandas as pd
import numpy as np
from typing import Tuple


class data_handle:
    irtdata_U_FNAME = "irtdata_U.csv"
    irtdata_Y_FNAME = "irtdata_Y.csv"
    irtdata_T_true_FNAME = "irtdata_T_true.csv"
    irtdata_icc_true_FNAME = "irtdata_icc_true.csv"
    irt_output_Y_FNAME = "irt_output_Y.csv"
    ICC_output_FNAME = "ICC_output.pdf"

    @classmethod
    def pandas_read(cls, indpath) -> Tuple:
        # パスを通す
        irtdata_U_fpath = os.path.join(indpath, cls.irtdata_U_FNAME)
        irtdata_Y_fpath = os.path.join(indpath, cls.irtdata_Y_FNAME)
        irtdata_T_true_fpath = os.path.join(indpath, cls.irtdata_T_true_FNAME)
        irtdata_icc_true_fpath = os.path.join(indpath, cls.irtdata_icc_true_FNAME)
        # csvから項目反応データを読み込む
        U_df = pd.read_csv(irtdata_U_fpath, header=None)
        # Yの初期クラス
        Y_df = pd.read_csv(irtdata_Y_fpath, header=None)
        # 正解クラス
        T_true_df = pd.read_csv(irtdata_T_true_fpath, header=None)
        # 真の項目反応曲線
        icc_true_df = pd.read_csv(irtdata_icc_true_fpath, header=None)
        return U_df, Y_df, T_true_df, icc_true_df

    @classmethod
    def output_result(cls, outdpath):
        # 結果の出力
        return 0

    @classmethod
    def df_to_array(cls, U_df, Y_df, T_true_df, icc_true_df) -> Tuple:
        # DataFrameからnp.arrayに変換
        U = U_df.values
        Y = Y_df.values
        T_true = T_true_df.values
        icc_true = icc_true_df.values
        I, J = np.shape(U)

        return U, Y, T_true, icc_true, I, J
