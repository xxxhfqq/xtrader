# 安装 akshare-proxy-patch 以保护东财接口，避免频繁调用被封禁
import akshare_proxy_patch
import os
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
_PATCH_HOST = os.environ.get("AKSHARE_PROXY_HOST", "").strip()
_PATCH_TOKEN = os.environ.get("AKSHARE_PROXY_TOKEN", "").strip()
_PATCH_RETRY = int(os.environ.get("AKSHARE_PROXY_RETRY", "30") or 30)
if _PATCH_HOST and _PATCH_TOKEN:
    akshare_proxy_patch.install_patch(_PATCH_HOST, _PATCH_TOKEN, _PATCH_RETRY)

import akshare as ak
import pandas as pd
from pathlib import Path
import numpy as np
import json

REQUIRED_INFER_COLUMNS = ["date", "code", "time", "open", "high", "low", "close", "volume"]

save_path = Path.cwd() / "data" #ratio 在config.json

def validate_akshare_data(df, code, function_name):
    """
    验证 akshare 获取的数据，检测 NaN 值和零值
    如果有问题则直接报错
    """
    if df is None or df.empty:
        raise ValueError(f"[{function_name}] akshare 返回空数据，code={code}")
    
    # 检查 NaN 值
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"[{function_name}] akshare 数据包含 NaN 值，code={code}, "
            f"包含 NaN 的列: {nan_cols}, NaN 数量: {df[nan_cols].isna().sum().to_dict()}"
        )
    
    # 检查数值列中的零值（对于价格和成交量等关键字段）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    price_volume_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['open', 'close', 'high', 'low', 'volume', '成交', '价', '量'])]
    
    zero_value_cols = []
    for col in price_volume_cols:
        if (df[col] == 0).any():
            zero_count = (df[col] == 0).sum()
            zero_value_cols.append(f"{col}({zero_count}个零值)")
    
    if zero_value_cols:
        raise ValueError(
            f"[{function_name}] akshare 数据包含零值，code={code}, "
            f"包含零值的列: {zero_value_cols}"
        )
    
    return df

def get_time(x):
    x = str(x)
    return x[0:10]
def get_time2(x):
    x = str(x)
    return np.int64(x[0:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16] + x[17:19] + "000")

def get_price_ak(code="sh.601988"):
    df1 = ak.stock_zh_a_hist_min_em(
        start_date=str(pd.to_datetime("today").date()),
        symbol=code[3:],
        period="1",
    )
    validate_akshare_data(df1, code, "get_price_ak")
    price = df1["收盘"].iloc[-1]
    if pd.isna(price) or price <= 0:
        raise ValueError(f"[get_price_ak] 获取的价格无效，code={code}, price={price}")
    return price

def get_bid_ask(code="sh.601988"):
    """
    从 akshare 获取买一/卖一价（buy_1 / sell_1），失败则返回 (None, None)
    如果数据包含 NaN 或零值，会直接报错
    """
    try:
        df = ak.stock_bid_ask_em(symbol=code[3:])
        validate_akshare_data(df, code, "get_bid_ask")
        # 最新接口字段示例：sell_1, buy_1, sell_1_vol, buy_1_vol, ...
        # 买一价 = buy_1, 卖一价 = sell_1
        row = df.set_index("item")["value"] if "item" in df.columns else df.iloc[:, 0:2].set_index(0)[1]
        ask = float(row.get("sell_1", 0))
        bid = float(row.get("buy_1", 0))
        if ask <= 0 or bid <= 0:
            raise ValueError(f"[get_bid_ask] 买一价或卖一价为0或负数，code={code}, bid={bid}, ask={ask}")
        if pd.isna(ask) or pd.isna(bid):
            raise ValueError(f"[get_bid_ask] 买一价或卖一价为 NaN，code={code}, bid={bid}, ask={ask}")
        return bid, ask
    except ValueError:
        # 重新抛出 ValueError（数据验证错误）
        raise
    except Exception as e:
        # 其他异常（如网络错误）返回 None, None
        return None, None
def update_infer_data_from_ak(code="sh.601988"):
    """
    从 akshare 更新 infer 数据，如果 akshare 返回空数据（可能是休市、非交易时段等），
    则从历史k线数据中提取最新数据更新 infer
    """
    try:
        with open(save_path / "config.json","r") as f:
            config = json.load(f)
            ratio = config.get(f"{code}_ratio", 1.0)
    except Exception:
        ratio = 1.0
    
    today_date = pd.to_datetime("today").date()
    end_date   = today_date.strftime("%Y%m%d")
    
    # 基于 infer.csv 的最后一条数据来决定从 akshare 拉取什么数据
    # 这样可以避免重复拉取，更精确地控制数据量
    infer_path = save_path / f"{code}_infer.csv"
    start_date = None
    last_infer_time = None  # infer.csv 最后一条数据的时间戳（用于过滤）
    
    if infer_path.exists():
        try:
            df_infer = pd.read_csv(infer_path)
            if len(df_infer) > 0:
                # 获取 infer.csv 最后一条数据的日期和时间
                last_infer_date_str = df_infer["date"].iloc[-1]
                last_infer_time_str = str(df_infer["time"].iloc[-1])
                
                # 解析日期
                last_infer_date = pd.to_datetime(last_infer_date_str).date()
                
                # 解析时间戳（格式：20260225150000000）
                if len(last_infer_time_str) >= 14:
                    last_infer_time = pd.to_datetime(last_infer_time_str[:14], format='%Y%m%d%H%M%S')
                
                # 无论最后一条数据是今天之前还是今天，都从今天开始拉取
                # 如果最后一条数据是今天的，后面会过滤掉早于该时间的数据
                start_date = today_date.strftime("%Y%m%d")
        except Exception as e:
            # 如果读取失败，使用昨天的日期作为备选
            start_date = (today_date - pd.Timedelta(days=1)).strftime("%Y%m%d")
    else:
        # infer.csv 不存在，从 BaoStock 数据获取倒数第二个交易日的日期（作为设计余量）
        csv_path = save_path / f"{code}.csv"
        if csv_path.exists():
            try:
                df_bs = pd.read_csv(csv_path)
                if len(df_bs) >= 60:
                    # 获取第-60条数据的日期（倒数第二个交易日）
                    second_last_date = pd.to_datetime(df_bs["date"].iloc[-60]).date()
                    start_date = second_last_date.strftime("%Y%m%d")
                else:
                    # 如果数据不足60条，使用倒数第二条数据的日期
                    if len(df_bs) >= 2:
                        second_last_date = pd.to_datetime(df_bs["date"].iloc[-2]).date()
                        start_date = second_last_date.strftime("%Y%m%d")
                    else:
                        # 如果数据很少，使用最后一条数据的日期
                        last_date = pd.to_datetime(df_bs["date"].iloc[-1]).date()
                        start_date = last_date.strftime("%Y%m%d")
            except Exception as e:
                # 如果读取失败，使用昨天的日期作为备选
                start_date = (today_date - pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            # 如果文件不存在，使用昨天的日期作为备选
            start_date = (today_date - pd.Timedelta(days=1)).strftime("%Y%m%d")
    
    # 尝试从 akshare 获取最新数据
    df1 = None
    try:
        df1 = ak.stock_zh_a_hist_min_em(
            symbol=code[3:],
            adjust="",
            period="5",
            start_date=start_date,
            end_date=end_date
        )
        # 如果返回空数据，则从历史数据中提取
        if df1 is None or df1.empty:
            df1 = None
        elif last_infer_time is not None:
            # 如果 infer.csv 的最后一条数据是今天的，过滤掉早于该时间的数据
            # 避免重复拉取已经存在的数据
            df1["time_parsed"] = pd.to_datetime(df1["时间"])
            df1 = df1[df1["time_parsed"] > last_infer_time].copy()
            if df1.empty:
                df1 = None
            else:
                df1 = df1.drop(columns=["time_parsed"])
    except Exception:
        df1 = None
    
    # 如果 akshare 返回有效数据，使用 akshare 数据
    if df1 is not None and not df1.empty:
        try:
            # 验证原始数据
            validate_akshare_data(df1, code, "update_infer_data_from_ak")
            
            df1 = df1.rename(columns={"时间":"time", "开盘":"open", "收盘":"close", "最高":"high", "最低":"low","成交量":"volume"})
            df1["date"] = df1["time"]
            df1["code"] = code
            df1["date"] = df1["date"].apply(get_time)
            df1["time"] = df1["time"].apply(get_time2)

            df1["volume"] = df1["volume"] * 100
            df1["open"]   = df1["open"] * ratio
            df1["close"]  = df1["close"] * ratio
            df1["high"]   = df1["high"] * ratio
            df1["low"]    = df1["low"] * ratio
            df1 = df1[["date", "code", "time", "open", "high", "low","close", "volume"]]
            
            # 验证处理后的数据
            validate_akshare_data(df1, code, "update_infer_data_from_ak(处理后)")
        except Exception as e:
            # 如果验证失败，使用历史数据
            print(f"[WARNING] akshare 数据验证失败 {code}: {e}，使用历史数据")
            df1 = None
    
    # 如果 akshare 没有数据或验证失败，从历史k线数据中提取最新数据
    if df1 is None or df1.empty:
        # 从历史k线数据中提取最近的数据
        csv_path = save_path / f"{code}.csv"
        if csv_path.exists():
            try:
                hist_df = pd.read_csv(csv_path)
                if len(hist_df) > 0:
                    # 获取最近2天的数据（约200-400条，足够用于推理）
                    recent_days = 2
                    hist_df["date_parsed"] = pd.to_datetime(hist_df["date"])
                    latest_date = hist_df["date_parsed"].max()
                    cutoff_date = latest_date - pd.Timedelta(days=recent_days)
                    df1 = hist_df[hist_df["date_parsed"] >= cutoff_date].copy()
                    if len(df1) > 0:
                        df1 = df1[["date", "code", "time", "open", "high", "low", "close", "volume"]]
                        # 不打印，避免日志过多（这些数据会合并到infer文件中，infer文件本身是4000条）
                    else:
                        df1 = None
            except Exception as e:
                print(f"[WARNING] 从历史数据提取失败 {code}: {e}")
                df1 = None
    
    # 更新 infer 文件：
    # 1) 以 x_bs 生成的 code_infer.csv 作为基线（该文件应是 code.csv 的末尾4k）
    # 2) 把 x_ak 拉到的最新细粒度数据合入基线
    # 3) 按 time 去重排序并截断到最近 4000 条
    infer_path = save_path / f"{code}_infer.csv"
    csv_path = save_path / f"{code}.csv"
    frames = []

    if infer_path.exists():
        try:
            base_infer_df = pd.read_csv(infer_path)
            if len(base_infer_df) > 0:
                base_infer_df = base_infer_df[REQUIRED_INFER_COLUMNS].copy()
                frames.append(base_infer_df)
        except Exception as e:
            print(f"[WARNING] 读取 infer 数据失败 {code}: {e}")

    # 兜底：若 infer 不存在，退化为使用历史K线末尾4k做基线。
    if not frames and csv_path.exists():
        try:
            hist_df = pd.read_csv(csv_path)
            if len(hist_df) > 0:
                hist_df = hist_df[REQUIRED_INFER_COLUMNS].copy()
                frames.append(hist_df.tail(4000))
        except Exception as e:
            print(f"[WARNING] 读取历史数据失败 {code}: {e}")

    if df1 is not None and not df1.empty:
        frames.append(df1[REQUIRED_INFER_COLUMNS].copy())

    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    merged["time"] = pd.to_numeric(merged["time"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["time"]).copy()
    merged["time"] = merged["time"].astype(np.int64)
    merged = merged.sort_values("time", ignore_index=True)
    merged = merged.drop_duplicates("time", keep="last")
    merged = merged.tail(4000).reset_index(drop=True)
    merged.to_csv(infer_path, index=False)



def set_ratio(code="sh.601988"):

    path1 = Path.cwd() / f"data/{code}.csv"  
    df1 = pd.read_csv(path1)
    df1 = df1[-1:]                   # 列: open

    df2 = ak.stock_zh_a_hist_min_em(symbol=code[3:],  
                                    start_date=pd.to_datetime("2026-01-20"),
                                    period="5",
                                    adjust="")        # 列: 开盘
    # 验证 akshare 数据
    validate_akshare_data(df2, code, "set_ratio")
    
    

    def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        给 df 统一补出 datetime 列，优先级：
        1) datetime
        2) day
        3) date + time
        - 如果 time 是类似 20190102093500000（YYYYMMDDHHMMSSxxx），取前14位解析
        - 如果 time 是类似 09:35:00，则和 date 拼成 'YYYY-MM-DD HH:MM:SS'
        """
        out = df.copy()

        if "date" in out.columns:
            out["time"] = pd.to_datetime(out["time"], format="%Y%m%d%H%M%S%f")
            out = out.rename(columns={"date":"No"})
            out = out.rename(columns={"time":"date"})

        if "时间" in out.columns:
            out = out.rename(columns={"时间": "date"}).copy()
            out["date"] = pd.to_datetime(out["date"])
            
            out["date"] = out["date"]

        return out


    df1 = ensure_datetime(df1)
    df2 = ensure_datetime(df2)

    m = df1.merge(df2, on="date")
    m["ratio1"] = m["open"] / m["开盘"]
    m["ratio2"] = m["close"] / m["收盘"]
    ratio = (m["ratio1"].iloc[0] + m["ratio2"].iloc[0]) / 2
    if not (save_path / "config.json").exists():
        (save_path / "config.json").touch()
    with open(save_path / "config.json", "r") as f:
        s = f.read().strip()
    a = json.loads(s) if s else {}
    with open(save_path / "config.json", "w") as f:
        a[f"{code}_ratio"] = ratio
        json.dump(a, f)
# set_ratio()
# update_infer_data_from_ak()


# stock_zh_a_minute返回字段
#                      day   open   high    low  close    volume
# 0     2025-11-21 14:55:00  6.290  6.290  6.270  6.290  11006419
# 1     2025-11-21 15:00:00  6.290  6.300  6.270  6.290  14472400

# def update_infer_data_from_ak(code="sh.601988"):
#     df1 = ak.stock_zh_a_minute(
#         symbol=code[0:2] + code[3:], #  symbol='sh000300'
#         period="5",                  #  period='1'; 获取 1, 5, 15, 30, 60 分钟的数据频率
#         adjust="hfq"                 #  adjust=""; 默认为空: 返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据;
#         )
    
#     df1 = df1.rename(columns={"day":"time"})
#     df1["date"] = df1["time"]
#     df1["code"] = code
#     df1["date"] = df1["date"].apply(get_time)
#     df1["time"] = df1["time"].apply(get_time2)

#     df1 = df1[-100:].reset_index(drop=True)
#     df2 = pd.read_csv(save_path / f"{code}_infer.csv")

#     df3 = pd.concat((df2, df1), ignore_index=True)
#     df3 = df3.drop_duplicates("time")
#     df3 = df3.sort_values("time", ignore_index=True)

#     df3.to_csv(save_path / f"{code}_infer.csv", index=False)

