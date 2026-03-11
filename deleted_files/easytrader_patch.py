"""
easytrader 验证码补丁模块

此模块用于动态修改 easytrader 的行为，以适应特定需求，例如处理验证码。
通过猴子补丁（Monkey Patching）的方式，在运行时替换原有方法。

要使用此补丁，请在您的主应用程序代码中，
在初始化 easytrader 的 Client 之前导入此模块：

import easytrader_patch

author: Doiiars
contact: doiiars@qq.com
date: 2025-07-23
"""
import re
import tempfile
from typing import List, Dict, Optional, TYPE_CHECKING
from io import StringIO

import pandas as pd
import pywinauto.keyboard
from easytrader.log import logger
from easytrader.utils.captcha import captcha_recognize
from easytrader.grid_strategies import Xls

if TYPE_CHECKING:
    from easytrader import clienttrader

def _xls_get_with_captcha(self: Xls, control_id: int) -> List[Dict]:
    """
    通过将 Grid 另存为 xls 文件再读取的方式获取 grid 内容
    此方法为 easytrader.grid_strategies.Xls.get 的补丁版本，增加了验证码处理逻辑。
    """
    grid = self._get_grid(control_id)

    # ctrl+s 保存 grid 内容为 xls 文件
    self._set_foreground(grid)
    grid.type_keys("^s", set_foreground=False)
    self._trader.wait(0.4)  # 等待“另存为”或验证码对话框弹出

    # 轮询识别当前是验证码框还是另存为（避免主程序负载时顶层仍是主窗口导致漏判验证码）
    captcha_win = None
    save_as_early = None
    for _ in range(10):  # 约 3 秒
        top_win = self._trader.app.top_window()
        if top_win:
            try:
                if top_win.window(class_name="Static", title_re="验证码").exists(timeout=0):
                    captcha_win = top_win
                    break
            except Exception:
                pass
            if not captcha_win and "另存为" in (top_win.window_text() or ""):
                save_as_early = top_win
                break
        if captcha_win or save_as_early:
            break
        # 顶层可能仍是主窗口，遍历所有窗口找验证码框
        try:
            for w in self._trader.app.windows():
                try:
                    if w.window(class_name="Static", title_re="验证码").exists(timeout=0):
                        captcha_win = w
                        break
                except Exception:
                    continue
            if captcha_win:
                break
        except Exception:
            pass
        self._trader.wait(0.3)

    # 检查并处理验证码（使用轮询得到的 captcha_win，避免漏判）
    if captcha_win is not None:
        top_win = captcha_win
        file_path = "tmp.png"
        count = 5
        found = False
        while count > 0:
            try:
                top_win.window(
                    control_id=0x965, class_name="Static"
                ).capture_as_image().save(file_path)
                
                captcha_num = captcha_recognize(file_path).strip()
                captcha_num = "".join(captcha_num.split())
                # 验证码只含字母大小写和数字，其他字符替换为 s
                captcha_num = re.sub(r"[^a-zA-Z0-9]", "s", captcha_num)
                logger.info(f"识别出的验证码: {captcha_num}")

                if len(captcha_num) == 4:
                    editor = top_win.window(control_id=0x964, class_name="Edit")
                    editor.select()
                    editor.type_keys(captcha_num, with_spaces=True)
                    
                    top_win.set_focus()
                    pywinauto.keyboard.send_keys("{ENTER}")
                    
                    # 验证码正确后，“另存为”对话框会成为新的顶层窗口
                    self._trader.wait(0.5)
                    if '另存为' in self._trader.app.top_window().window_text():
                        found = True
                        logger.info("验证码正确，继续执行保存操作。")
                        break
                    logger.warning(f"验证码 {captcha_num} 错误或无法通过验证。")
                else:
                    logger.warning(f"验证码识别结果非4位: {captcha_num}")

            except Exception as e:
                logger.error(f"处理验证码时发生异常: {e}")
            
            count -= 1
            self._trader.wait(0.4)
            try:
                top_win.window(control_id=0x965, class_name="Static").click()
            except Exception as e:
                logger.error(f"刷新验证码时出错: {e}")
                break
        
        if not found:
            logger.error("验证码处理失败，取消操作。")
            try:
                top_win.Button2.click()  # 点击取消
            except Exception as e:
                logger.error(f"点击取消按钮时出错: {e}")
            return []

    # 等待“另存为”窗口成为顶层（无验证码时弹窗可能稍慢；或轮询时已得到则直接用）
    save_as_window = save_as_early
    if save_as_window is None:
        for _ in range(12):  # 最多约 3 秒
            w = self._trader.app.top_window()
            if w and '另存为' in (w.window_text() or ""):
                save_as_window = w
                break
            self._trader.wait(0.25)
    if save_as_window is None:
        top_win = self._trader.app.top_window()
        logger.error(f"未能定位到'另存为'对话框，当前顶层窗口为: {top_win.window_text() if top_win else 'None'}")
        return []

    temp_path = tempfile.mktemp(suffix=".xls", dir=self.tmp_folder)
    self._set_foreground(save_as_window)

    save_as_window.Edit1.set_edit_text(temp_path)
    self._trader.wait(0.1)
    save_as_window.type_keys("%{s}%{y}", set_foreground=False)
    
    self._trader.wait(0.5)
    
    if self._trader.is_exist_pop_dialog():
        self._trader.app.top_window().Button2.click()
        self._trader.wait(0.4)
    
    # 从Xls类中调用原始的_format_grid_data方法
    return Xls._format_grid_data(self, temp_path)


def _format_grid_data(self, data: str) -> List[Dict]:
    """
    从xls文件中读取并格式化数据。
    这个辅助函数被添加到补丁中，以防原始类结构发生变化。
    """
    with open(data, encoding="gbk", errors="replace") as f:
        content = f.read()

    df = pd.read_csv(
        StringIO(content),
        delimiter="\t",
        dtype=self._trader.config.GRID_DTYPE,
        na_filter=False,
    )
    return df.to_dict("records")


def _patch_easytrader():
    """应用补丁"""
    if not hasattr(Xls, '_get_original'):
        Xls._get_original = Xls.get
        Xls.get = _xls_get_with_captcha
        # 同时将格式化函数也加入，增强兼容性
        if not hasattr(Xls, '_format_grid_data'):
            Xls._format_grid_data = _format_grid_data
        logger.info("成功应用 easytrader Xls 策略的验证码处理补丁 (已修正)。")

# 执行补丁
_patch_easytrader() 