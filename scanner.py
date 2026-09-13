import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

import pandas as pd
import yfinance as yf


# ============================================================
# 1. 可调参数
# ============================================================

# 前高 A 左右各需要多少根K确认它是历史波峰
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# A 至少要是当时最近多少日内比较突出的高点
A_CONTEXT_BARS = 30

# A 后面至少回撤多少
MIN_PULLBACK_PCT = 8.0

# A 与今天之间至少间隔多少个交易日
MIN_DAYS_AFTER_A = 5

# 最多往前寻找多少个交易日的 A
MAX_DAYS_AFTER_A = 100

# “再次冲击前高”的价格区间
#
# 例如：
# A = 100
# 下方6% -> 94
# 上方8% -> 108
#
# 今天的价格只要重新进入 94~108 这个区域，
# 就属于“重新冲击前高”的候选。
RETEST_BELOW_PCT = 6.0
RETEST_ABOVE_PCT = 8.0

# 为了证明价格确实曾经离开过前高区域，
# 回撤后至少要有多少天 high 低于前高区域下沿。
MIN_DAYS_AWAY = 3

# A 之前多少天内至少要有一定的上涨推动
IMPULSE_LOOKBACK = 15
MIN_IMPULSE_PCT = 4.0

# 下载多少历史数据
DOWNLOAD_PERIOD = "1y"


# ============================================================
# 2. 获取 Nasdaq-100 股票列表
# ============================================================

def get_nasdaq100_symbols():
    """
    从 Wikipedia 自动读取 Nasdaq-100 当前成分股。
    不需要你手工维护100只股票代码。
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"

    tables = pd.read_html(url)

    for table in tables:
        if "Ticker" in table.columns:
            symbols = table["Ticker"].astype(str).tolist()

            # Yahoo Finance 中 BRK.B 这类格式一般使用 BRK-B。
            symbols = [s.replace(".", "-").strip() for s in symbols]

            if len(symbols) >= 90:
                return symbols

    raise RuntimeError("无法从 Wikipedia 获取 Nasdaq-100 成分股列表。")


# ============================================================
# 3. 判断某一天是不是历史波峰 A
# ============================================================

def is_pivot_high(highs, i):
    """
    A 是历史波峰，因此允许使用它之后已经发生的几根K来确认。
    这不会造成未来函数，因为扫描的是“今天”，A已经在历史中。
    """
    if i < PIVOT_LEFT:
        return False

    if i + PIVOT_RIGHT >= len(highs):
        return False

    center = highs.iloc[i]

    left = highs.iloc[i - PIVOT_LEFT:i]
    right = highs.iloc[i + 1:i + PIVOT_RIGHT + 1]

    return center >= left.max() and center > right.max()


# ============================================================
# 4. 判断 A 是否是相对明显的高点
# ============================================================

def is_major_high(df, i):
    start = max(0, i - A_CONTEXT_BARS + 1)

    context_high = df["High"].iloc[start:i + 1].max()
    a_price = df["High"].iloc[i]

    # A至少处于这段时间最高区域的2%以内
    if a_price < context_high * 0.98:
        return False

    # A以前应该有一定上涨推动
    impulse_start = max(0, i - IMPULSE_LOOKBACK)

    impulse_low = df["Low"].iloc[impulse_start:i + 1].min()

    if impulse_low <= 0:
        return False

    impulse_pct = (a_price - impulse_low) / impulse_low * 100

    return impulse_pct >= MIN_IMPULSE_PCT


# ============================================================
# 5. 检查一只股票今天是否“再次冲击前高”
# ============================================================

def scan_symbol(symbol):
    try:
        df = yf.download(
            symbol,
            period=DOWNLOAD_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or len(df) < 60:
            return None

        # yfinance 某些版本会产生 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        ).copy()

        if len(df) < 60:
            return None

        today_i = len(df) - 1

        today_high = float(df["High"].iloc[today_i])
        today_low = float(df["Low"].iloc[today_i])
        today_close = float(df["Close"].iloc[today_i])

        # --------------------------------------------
        # 从距离今天较近的历史波峰开始向前寻找
        #
        # 我们寻找的是：
        #
        # A
        # ↓
        # 明显回撤
        # ↓
        # 今天第一次重新进入A附近
        # --------------------------------------------

        earliest_a = max(
            PIVOT_LEFT,
            today_i - MAX_DAYS_AFTER_A
        )

        latest_a = today_i - MIN_DAYS_AFTER_A

        if latest_a <= earliest_a:
            return None

        candidate_indices = list(
            range(earliest_a, latest_a + 1)
        )

        # 最近的合格A优先
        candidate_indices.reverse()

        for a_i in candidate_indices:

            # ----------------------------------------
            # A必须是历史明确波峰
            # ----------------------------------------
            if not is_pivot_high(df["High"], a_i):
                continue

            if not is_major_high(df, a_i):
                continue

            a_price = float(df["High"].iloc[a_i])

            if a_price <= 0:
                continue

            # ----------------------------------------
            # A以后，到昨天为止
            #
            # 今天不能参与决定“过去有没有回撤”，
            # 防止逻辑混乱。
            # ----------------------------------------
            after_a = df.iloc[a_i + 1:today_i]

            if len(after_a) < MIN_DAYS_AFTER_A - 1:
                continue

            c_price = float(after_a["Low"].min())
            c_date = after_a["Low"].idxmin()

            pullback_pct = (
                (a_price - c_price)
                / a_price
                * 100
            )

            # ----------------------------------------
            # 中间必须真正发生明显回撤
            # ----------------------------------------
            if pullback_pct < MIN_PULLBACK_PCT:
                continue

            # ----------------------------------------
            # 定义A附近的“再次冲击区”
            # ----------------------------------------
            lower_band = (
                a_price *
                (1 - RETEST_BELOW_PCT / 100)
            )

            upper_band = (
                a_price *
                (1 + RETEST_ABOVE_PCT / 100)
            )

            # ----------------------------------------
            # 必须真的离开过A附近
            #
            # 至少若干天的最高价都低于下沿，
            # 避免高位横盘被误认为“再次冲击”。
            # ----------------------------------------
            away_days = (
                after_a["High"] < lower_band
            ).sum()

            if away_days < MIN_DAYS_AWAY:
                continue

            # ----------------------------------------
            # 今天必须进入A附近区域
            #
            # 使用整根K的价格范围是否与区域相交，
            # 这样跳空上涨也能识别。
            # ----------------------------------------
            today_enters_zone = (
                today_high >= lower_band
                and today_low <= upper_band
            )

            if not today_enters_zone:
                continue

            # ----------------------------------------
            # 关键：
            # 今天应该是“重新进入”的第一天。
            #
            # 如果昨天已经在A附近，
            # 今天就不重复提醒。
            # ----------------------------------------
            yesterday_high = float(
                df["High"].iloc[today_i - 1]
            )

            yesterday_low = float(
                df["Low"].iloc[today_i - 1]
            )

            yesterday_in_zone = (
                yesterday_high >= lower_band
                and yesterday_low <= upper_band
            )

            if yesterday_in_zone:
                continue

            # ----------------------------------------
            # 最好是从下方向上重新攻击
            #
            # 如果昨天已经明显高于整个A区，
            # 就不是我们要的“从下方重新冲击”。
            # ----------------------------------------
            if yesterday_high > upper_band:
                continue

            distance_pct = (
                (today_high - a_price)
                / a_price
                * 100
            )

            a_date = df.index[a_i]

            bars_from_a = today_i - a_i

            return {
                "symbol": symbol,
                "a_date": str(a_date.date()),
                "a_price": a_price,
                "c_date": str(c_date.date()),
                "c_price": c_price,
                "pullback_pct": pullback_pct,
                "today_high": today_high,
                "today_close": today_close,
                "distance_pct": distance_pct,
                "bars_from_a": bars_from_a,
                "zone_low": lower_band,
                "zone_high": upper_band,
            }

        return None

    except Exception as exc:
        print(f"[ERROR] {symbol}: {exc}")
        return None


# ============================================================
# 6. 批量扫描 Nasdaq-100
# ============================================================

def run_scan():
    symbols = get_nasdaq100_symbols()

    print(f"Nasdaq-100 symbols: {len(symbols)}")

    results = []

    for idx, symbol in enumerate(symbols, start=1):

        print(
            f"[{idx}/{len(symbols)}] "
            f"Scanning {symbol}..."
        )

        result = scan_symbol(symbol)

        if result is not None:
            results.append(result)

    return results


# ============================================================
# 7. 生成邮件
# ============================================================

def build_email(results):
    today = datetime.now().strftime("%Y-%m-%d")

    if not results:
        subject = (
            f"Nasdaq-100 前高再次冲击扫描 "
            f"{today}：无候选"
        )

        body = (
            f"{today}\n\n"
            "今天 Nasdaq-100 没有发现符合条件的"
            "“明显回撤后再次冲击前高”股票。\n"
        )

        return subject, body

    results = sorted(
        results,
        key=lambda x: abs(x["distance_pct"])
    )

    subject = (
        f"Nasdaq-100 前高再次冲击提醒 "
        f"{today}：{len(results)} 只"
    )

    lines = []

    lines.append(
        f"{today} Nasdaq-100 扫描结果"
    )

    lines.append("")
    lines.append(
        "以下股票今天重新进入历史主波峰附近。"
    )

    lines.append(
        "本扫描只做价格结构初筛，不代表买卖建议。"
    )

    lines.append("")

    for r in results:

        lines.append(
            "================================"
        )

        lines.append(
            f"股票：{r['symbol']}"
        )

        lines.append(
            f"前高 A 日期：{r['a_date']}"
        )

        lines.append(
            f"前高 A：{r['a_price']:.2f}"
        )

        lines.append(
            f"中间最低 C 日期：{r['c_date']}"
        )

        lines.append(
            f"中间最低 C：{r['c_price']:.2f}"
        )

        lines.append(
            f"A→C 最大回撤："
            f"{r['pullback_pct']:.2f}%"
        )

        lines.append(
            f"今天最高价：{r['today_high']:.2f}"
        )

        lines.append(
            f"今天收盘价：{r['today_close']:.2f}"
        )

        lines.append(
            f"今天最高价相对A："
            f"{r['distance_pct']:+.2f}%"
        )

        lines.append(
            f"前高附近区域："
            f"{r['zone_low']:.2f} ~ "
            f"{r['zone_high']:.2f}"
        )

        lines.append(
            f"A距今天：{r['bars_from_a']} 根日K"
        )

        lines.append("")

    return subject, "\n".join(lines)


# ============================================================
# 8. 发邮件
# ============================================================

def send_email(subject, body):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(
        os.getenv("SMTP_PORT", "465")
    )

    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    email_to = os.getenv("EMAIL_TO")

    missing = []

    if not smtp_host:
        missing.append("SMTP_HOST")

    if not smtp_user:
        missing.append("SMTP_USER")

    if not smtp_pass:
        missing.append("SMTP_PASS")

    if not email_to:
        missing.append("EMAIL_TO")

    if missing:
        raise RuntimeError(
            "缺少 GitHub Secrets: "
            + ", ".join(missing)
        )

    msg = MIMEText(
        body,
        "plain",
        "utf-8"
    )

    msg["Subject"] = Header(
        subject,
        "utf-8"
    )

    msg["From"] = smtp_user
    msg["To"] = email_to

    # 465通常使用SSL
    if smtp_port == 465:

        with smtplib.SMTP_SSL(
            smtp_host,
            smtp_port,
            timeout=30
        ) as server:

            server.login(
                smtp_user,
                smtp_pass
            )

            server.sendmail(
                smtp_user,
                [email_to],
                msg.as_string()
            )

    # 587通常使用STARTTLS
    else:

        with smtplib.SMTP(
            smtp_host,
            smtp_port,
            timeout=30
        ) as server:

            server.starttls()

            server.login(
                smtp_user,
                smtp_pass
            )

            server.sendmail(
                smtp_user,
                [email_to],
                msg.as_string()
            )


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Nasdaq-100 "
        "retest scanner..."
    )

    results = run_scan()

    print("")
    print(
        f"Found {len(results)} candidate(s)."
    )

    for r in results:
        print(
            r["symbol"],
            f"A={r['a_price']:.2f}",
            f"Pullback={r['pullback_pct']:.2f}%",
            f"TodayHigh={r['today_high']:.2f}",
            f"Distance={r['distance_pct']:+.2f}%"
        )

    subject, body = build_email(results)

    send_email(
        subject,
        body
    )

    print("Email sent successfully.")
