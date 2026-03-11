import baostock as bs
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# 安装 akshare-proxy-patch 以保护东财接口，避免频繁调用被封禁
import akshare_proxy_patch
akshare_proxy_patch.install_patch("101.201.173.125", "20260226NX2E", 30)  # AUTH_TOKEN已配置，retry=30表示重试次数

import akshare as ak
import numpy as np
from datetime import datetime, timedelta
save_path = Path.cwd() / "data"
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

# ==================== 重试配置 ====================
# 重试配置
MAX_RETRIES = 3      # 最大重试次数

# 批量拉取配置
BATCH_SIZE = 30      # 每批拉取的天数（减少请求次数）

def _fetch_days_individually(code, fields, batch_start, batch_end, frequency, adjustflag):
    """
    逐天拉取数据（当批量请求失败时的备用方案）
    """
    batch_data_list = []
    if isinstance(batch_start, pd.Timestamp):
        date_range = pd.date_range(start=batch_start.date(), end=batch_end.date())
    else:
        date_range = pd.date_range(start=batch_start, end=batch_end)
    
    for date in date_range:
        retry_count = 0
        success = False
        
        while retry_count <= MAX_RETRIES and not success:
            try:
                date_str = str(date.date())
                rs = bs.query_history_k_data_plus(
                    code=code,
                    fields=fields,
                    start_date=date_str,
                    end_date=date_str,
                    frequency=frequency,
                    adjustflag=adjustflag
                )
                
                if rs.error_code == "0":
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        batch_data_list.append(pd.DataFrame(data_list, columns=rs.fields))
                    
                    success = True
                else:
                    if retry_count < MAX_RETRIES:
                        retry_count += 1
                    else:
                        print(f"[WARNING] 跳过日期 {date_str}，已达最大重试次数")
                        break
                        
            except Exception as e:
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                else:
                    print(f"[WARNING] 跳过日期 {date_str}，异常: {e}")
                    break
    
    return batch_data_list


def update_data_from_bs(code="sh.601988", verbose=False):
    """
    拉取bs最新的历史数据,晚上八点后更新当日股票复权
    调用这个之后，会自动把后四千行写入infer，用作结合实盘最新数据做推断
    
    数据验证：
    - 如果起始日期和sh.603323.csv一样，新拉取的数据行数应该至少和sh.603323.csv一样长
    - 精确检查缺失的具体日期
    - 如果有问题，则删掉拉取的数据重新拉取
    
    Args:
        code: 股票代码
        verbose: 是否打印详细信息，默认为False（静默模式）
    """
    path = save_path / f"{code}.csv"
    reference_path = save_path / "sh.603323.csv"
    
    if not path.exists(): # 如果不存在,则从头开始拉取数据
        get_data_from_bs(code=code)
        return None
    
    df = pd.read_csv(path)
    start_date = pd.to_datetime(df["date"].iloc[-1]).date()# + pd.Timedelta(days=1)
    end_date   = pd.to_datetime("today").date()
    
    # 获取参考文件（sh.603323.csv）的起始日期、行数和日期集合
    reference_start_date = None
    reference_row_count = None
    reference_dates = set()
    if reference_path.exists():
        try:
            ref_df = pd.read_csv(reference_path)
            if len(ref_df) > 0:
                reference_start_date = pd.to_datetime(ref_df["date"].iloc[0]).date()
                reference_row_count = len(ref_df)
                # 提取参考文件的所有日期（只取日期部分，忽略时间）
                reference_dates = set(pd.to_datetime(ref_df["date"]).dt.date.unique())
                if verbose:
                    print(f"[INFO] 参考文件 sh.603323.csv: 起始日期={reference_start_date}, 行数={reference_row_count}, 唯一日期数={len(reference_dates)}")
        except Exception as e:
            if verbose:
                print(f"[WARNING] 读取参考文件失败: {e}")
    
    # 拉取新数据
    get_data_from_bs(code=code, start_date=start_date, end_date=end_date, merge=True, df=df, verbose=verbose)
    
    # 验证新拉取的数据
    if reference_path.exists() and reference_start_date is not None and reference_row_count is not None:
        try:
            # 重新读取更新后的数据
            new_df = pd.read_csv(path)
            if len(new_df) > 0:
                new_start_date = pd.to_datetime(new_df["date"].iloc[0]).date()
                new_row_count = len(new_df)
                
                # 提取当前文件的所有日期（只取日期部分，忽略时间）
                new_dates = set(pd.to_datetime(new_df["date"]).dt.date.unique())
                
                if verbose:
                    print(f"[INFO] 新拉取的数据 {code}.csv: 起始日期={new_start_date}, 行数={new_row_count}, 唯一日期数={len(new_dates)}")
                
                # 准备用于验证的参考数据
                # 如果起始日期不同，从参考文件中截取与被验证数据相同起始日期的数据
                if new_start_date != reference_start_date:
                    if verbose:
                        print(f"[INFO] 起始日期不同，从参考文件中截取从 {new_start_date} 开始的数据进行验证")
                    try:
                        # 重新读取参考文件
                        ref_df_full = pd.read_csv(reference_path)
                        # 筛选出从 new_start_date 开始的数据
                        ref_df_filtered = ref_df_full[pd.to_datetime(ref_df_full["date"]).dt.date >= new_start_date]
                        if len(ref_df_filtered) > 0:
                            # 更新参考数据
                            reference_start_date = new_start_date
                            reference_row_count = len(ref_df_filtered)
                            reference_dates = set(pd.to_datetime(ref_df_filtered["date"]).dt.date.unique())
                            if verbose:
                                print(f"[INFO] 参考数据（截取后）: 起始日期={reference_start_date}, 行数={reference_row_count}, 唯一日期数={len(reference_dates)}")
                        else:
                            if verbose:
                                print(f"[WARNING] 参考文件中没有从 {new_start_date} 开始的数据，跳过验证")
                            return None
                    except Exception as e:
                        if verbose:
                            print(f"[WARNING] 截取参考数据失败: {e}，使用原始参考数据")
                
                # 如果起始日期相同（或已截取为相同），检查行数和缺失日期
                if new_start_date == reference_start_date:
                    # 计算缺失的数据条数
                    missing_row_count = reference_row_count - new_row_count
                    
                    # 找出缺失的日期
                    missing_dates = reference_dates - new_dates
                    
                    # 只有当缺失数据超过1500条时才重新拉取
                    if missing_row_count > 1500:
                        # 将缺失日期排序并格式化
                        missing_dates_sorted = sorted(list(missing_dates)) if missing_dates else []
                        missing_dates_str = [str(d) for d in missing_dates_sorted]
                        
                        print(f"[WARNING] {code} 数据验证失败: 缺失 {missing_row_count} 条数据（超过1500条阈值），重新拉取")
                        if verbose and missing_dates:
                            print(f"[WARNING] 缺失 {len(missing_dates)} 个日期的数据")
                            print(f"[WARNING] 缺失的日期: {', '.join(missing_dates_str[:20])}")  # 只显示前20个
                            if len(missing_dates) > 20:
                                print(f"[WARNING] ... 还有 {len(missing_dates) - 20} 个日期缺失")
                        
                        # 删除有问题的文件
                        if path.exists():
                            path.unlink()
                        infer_path = save_path / f"{code}_infer.csv"
                        if infer_path.exists():
                            infer_path.unlink()
                        
                        # 重新拉取数据（从头开始，不合并）
                        get_data_from_bs(code=code, start_date=reference_start_date, end_date=end_date, merge=False, verbose=verbose)
                        
                        # 再次验证
                        verify_df = pd.read_csv(path)
                        verify_row_count = len(verify_df)
                        verify_dates = set(pd.to_datetime(verify_df["date"]).dt.date.unique())
                        verify_missing_row_count = reference_row_count - verify_row_count
                        verify_missing_dates = reference_dates - verify_dates
                        
                        if verify_missing_row_count > 1500:
                            verify_missing_dates_sorted = sorted(list(verify_missing_dates)) if verify_missing_dates else []
                            verify_missing_dates_str = [str(d) for d in verify_missing_dates_sorted]
                            print(f"[WARNING] {code} 重新拉取后仍缺失 {verify_missing_row_count} 条数据（超过1500条阈值）")
                            if verbose and verify_missing_dates:
                                print(f"[WARNING] 缺失 {len(verify_missing_dates)} 个日期的数据")
                                print(f"[WARNING] 缺失的日期: {', '.join(verify_missing_dates_str[:20])}")
                                if len(verify_missing_dates) > 20:
                                    print(f"[WARNING] ... 还有 {len(verify_missing_dates) - 20} 个日期缺失")
                            print(f"[WARNING] 可能需要手动检查数据源或网络连接")
                        # else: 验证通过，不打印
                    elif missing_row_count > 0:
                        # 缺失数据在1500条以内，只在verbose模式下打印
                        if verbose:
                            missing_dates_sorted = sorted(list(missing_dates)) if missing_dates else []
                            missing_dates_str = [str(d) for d in missing_dates_sorted]
                            print(f"[INFO] 数据验证: {code}.csv 缺失 {missing_row_count} 条数据（在允许范围内，不重新拉取）")
                            if missing_dates:
                                print(f"[INFO] 缺失 {len(missing_dates)} 个日期的数据")
                                print(f"[INFO] 缺失的日期: {', '.join(missing_dates_str[:20])}")  # 只显示前20个
                                if len(missing_dates) > 20:
                                    print(f"[INFO] ... 还有 {len(missing_dates) - 20} 个日期缺失")
                    # else: 验证通过，不打印
                else:
                    # 如果起始日期仍然不同（截取参考数据失败的情况），只在verbose模式下打印
                    if verbose:
                        print(f"[INFO] 起始日期不同且无法截取参考数据，跳过验证: {code} 起始日期={new_start_date}, 参考起始日期={reference_start_date}")
        except Exception as e:
            print(f"[ERROR] 验证数据时出错: {e}")



def get_data_from_bs(code="sh.601988", fields="date,time,code,open,high,low,close,volume", start_date="2019-01-01", end_date=None, frequency="5", adjustflag="1", merge=False, df=None, verbose=False):
    """
    从baostock拉取数据，全速下载：
    - 批量拉取：减少请求次数
    - 错误重试：自动重试失败的请求
    
    Args:
        verbose: 是否打印详细信息，默认为False（静默模式）
    """
    total_data_list = []
    
    # 登录（保持会话）
    lg = bs.login()
    if lg.error_code != "0":
        print(f"[ERROR] baostock登录失败: {lg.error_msg}")
        return
    
    if end_date is None:
        end_date = str(pd.to_datetime("today").date())
    
    # 将start_date转换为字符串格式（如果还不是）
    if isinstance(start_date, pd.Timestamp):
        start_date = str(start_date.date())
    elif hasattr(start_date, 'date'):
        start_date = str(start_date)
    
    date_range = pd.date_range(start=start_date, end=end_date)
    total_days = len(date_range)
    
    # 策略选择：如果天数较少，直接批量拉取；如果天数较多，分批拉取
    if total_days <= BATCH_SIZE:
        # 天数较少，直接一次性拉取
        if verbose:
            print(f"[INFO] 总共 {total_days} 天数据，一次性拉取")
        batches = [(date_range[0], date_range[-1])]
    else:
        # 天数较多，分成多个批次
        batches = []
        for i in range(0, total_days, BATCH_SIZE):
            batch_start_idx = i
            batch_end_idx = min(i + BATCH_SIZE - 1, total_days - 1)
            batches.append((date_range[batch_start_idx], date_range[batch_end_idx]))
        if verbose:
            print(f"[INFO] 总共 {total_days} 天数据，分成 {len(batches)} 批拉取（每批最多 {BATCH_SIZE} 天）")
    
    # 只在verbose模式下显示进度条，否则静默执行
    pbar = tqdm(total=len(batches), desc=f"拉取 {code} 数据", disable=not verbose, leave=False)
    for batch_idx, (batch_start, batch_end) in enumerate(batches):
            retry_count = 0
            success = False
            batch_data_list = []
            
            while retry_count <= MAX_RETRIES and not success:
                try:
                    batch_start_str = str(batch_start.date())
                    batch_end_str = str(batch_end.date())
                    
                    # 批量请求数据（一次请求多天）
                    rs = bs.query_history_k_data_plus(
                        code=code,
                        fields=fields,
                        start_date=batch_start_str,
                        end_date=batch_end_str,
                        frequency=frequency,
                        adjustflag=adjustflag
                    )
                    
                    # 检查错误码
                    if rs.error_code != "0":
                        error_msg = rs.error_msg
                        
                        # 判断是否被拉黑或限流
                        if "频繁" in error_msg or "限制" in error_msg or "IP" in error_msg or "请求过多" in error_msg:
                            if verbose:
                                print(f"\n[WARNING] 疑似被限流或拉黑: {error_msg}")
                            retry_count += 1
                            continue
                        
                        # 其他错误
                        if retry_count < MAX_RETRIES:
                            if verbose:
                                print(f"\n[WARNING] 批次请求失败 (尝试 {retry_count + 1}/{MAX_RETRIES + 1}): {error_msg}")
                                print(f"[INFO] 批次范围: {batch_start_str} 到 {batch_end_str}")
                            retry_count += 1
                            continue
                        else:
                            print(f"\n[ERROR] 批次请求失败，已达最大重试次数: {error_msg}")
                            print(f"[ERROR] 批次范围: {batch_start_str} 到 {batch_end_str}")
                            # 如果批量请求失败，尝试逐天拉取
                            print(f"[INFO] 尝试逐天拉取该批次数据...")
                            batch_data_list = _fetch_days_individually(code, fields, batch_start, batch_end, frequency, adjustflag)
                            success = True
                            break
                    
                    # 读取数据
                    data_list = []
                    while rs.error_code == "0" and rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        batch_data_list = [pd.DataFrame(data_list, columns=rs.fields)]
                    
                    success = True
                    if verbose:
                        pbar.set_postfix({"批次": f"{batch_idx + 1}/{len(batches)}", "范围": f"{batch_start_str}~{batch_end_str}"})
                    
                except Exception as e:
                    if retry_count < MAX_RETRIES:
                        if verbose:
                            print(f"\n[WARNING] 批次请求异常 (尝试 {retry_count + 1}/{MAX_RETRIES + 1}): {e}")
                        retry_count += 1
                    else:
                        print(f"[ERROR] {code} 批次请求异常，已达最大重试次数: {e}")
                        if verbose:
                            print(f"[INFO] 尝试逐天拉取该批次数据...")
                        # 如果批量请求失败，尝试逐天拉取
                        batch_data_list = _fetch_days_individually(code, fields, batch_start, batch_end, frequency, adjustflag)
                        success = True
                        break
            
            # 收集批次数据
            if batch_data_list:
                total_data_list.extend(batch_data_list)
            pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 处理数据（无论verbose与否都需要执行）
    if total_data_list:
        data = pd.concat(total_data_list, ignore_index=True)
        data['time'] = data["time"].astype(np.int64) # 要小心, 有时候是int有时候是str
        if not merge:
            data = data.sort_values("time", ignore_index=True)
            data = data.drop_duplicates(subset="time", keep="last")
            data.to_csv(save_path / f"{code}.csv", index=False)
        if merge:
            df["time"] = df["time"].astype(np.int64)
            data = pd.concat((df, data), ignore_index=True)
            data = data.sort_values("time", ignore_index=True)
            data = data.drop_duplicates(subset="time", keep="last")
            data.to_csv(save_path / f"{code}.csv", index=False)
    
    # 登出
    lg = bs.logout()
    if lg.error_code != "0" and verbose:
        print(f"[WARNING] baostock登出失败: {lg.error_msg}")
    
    # 读取数据文件生成infer文件（如果文件存在）
    csv_path = save_path / f"{code}.csv"
    if csv_path.exists():
        data = pd.read_csv(csv_path)
        if len(data) > 0:
            infer_data = data.iloc[-4000:, :]
            infer_data.to_csv(save_path / f"{code}_infer.csv", index=False)
        else:
            if verbose:
                print(f"[WARNING] {code}.csv 文件为空，无法生成 infer 文件")
    else:
        if verbose:
            print(f"[WARNING] {code}.csv 文件不存在，无法生成 infer 文件")

# update_data_from_bs()

