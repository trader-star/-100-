import os
import smtplib
from io import StringIO
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests


# ============================================================
# 1. 可调参数
# ============================================================

# 历史波峰识别
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# A必须是最近一段时间比较明显的高点
A_CONTEXT_BARS = 30

# A之后至少回撤多少
MIN_PULLBACK_PCT = 8.0

# A距离今天至少多少个交易日
MIN_DAYS_AFTER_A = 5

# 最多寻找多少个交易日前的A
MAX_DAYS_AFTER_A = 100

# 今天重新接近A的范围
#
# 例如A = 100：
# 下方6% = 94
# 上方8% = 108
#
# 只要今天重新进入94~108，就进入候选范围
RETEST_BELOW_PCT = 6.0
RETEST_ABOVE_PCT = 8.0

# 中间必须真正离开过前高区域
# 至少多少个交易日的最高价低于A区域下沿
MIN_DAYS_AWAY = 3

# A之前的上涨推动
IMPULSE_LOOKBACK = 15
MIN_IMPULSE_PCT = 4.0

# 下载历史数据长度
DOWNLOAD_PERIOD = "1y"


# ============================================================
# 2. 获取 Nasdaq-100 当前成分股
# ============================================================

def get_nasdaq100_symbols():

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"

    # 模拟正常浏览器访问，避免Wikipedia返回403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    for table in tables:

        # 有时候网页表头可能是MultiIndex
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                str(col[-1]).strip()
                for col in table.columns
            ]

        columns = [
            str(c).strip()
            for c in table.columns
        ]

        table.columns = columns

        # Wikipedia通常叫Ticker
        if "Ticker" in table.columns:

            symbols = (
                table["Ticker"]
                .astype(str)
                .str.strip()
                .tolist()
            )

            symbols = [
                s.replace(".", "-")
                for s in symbols
            ]

            if len(symbols) >= 90:
                print(
                    f"Successfully loaded "
                    f"{len(symbols)} Nasdaq-100 symbols."
                )
                return symbols

        # 防止未来表头改成Symbol
        if "Symbol" in table.columns:

            symbols = (
                table["Symbol"]
                .astype(str)
                .str.strip()
                .tolist()
            )

            symbols = [
                s.replace(".", "-")
                for s in symbols
            ]

            if len(symbols) >= 90:
                print(
                    f"Successfully loaded "
                    f"{len(symbols)} Nasdaq-100 symbols."
                )
                return symbols

    raise RuntimeError(
        "Nasdaq-100 component table was not found."
    )


# ============================================================
# 3. 判断历史某根K是否为明确波峰
# ============================================================

def is_pivot_high(highs, i):

    if i < PIVOT_LEFT:
        return False

    if i + PIVOT_RIGHT >= len(highs):
        return False

    center = float(highs.iloc[i])

    left = highs.iloc[
        i - PIVOT_LEFT:i
    ]

    right = highs.iloc[
        i + 1:i + PIVOT_RIGHT + 1
    ]

    if len(left) == 0 or len(right) == 0:
        return False

    return (
        center >= float(left.max())
        and
        center > float(right.max())
    )


# ============================================================
# 4. 判断A是不是比较明显的历史高点
# ============================================================

def is_major_high(df, i):

    start = max(
        0,
        i - A_CONTEXT_BARS + 1
    )

    context_high = float(
        df["High"]
        .iloc[start:i + 1]
        .max()
    )

    a_price = float(
        df["High"].iloc[i]
    )

    # A至少位于近期最高区域2%以内
    if a_price < context_high * 0.98:
        return False

    # 检查A之前是否存在一定上涨推动
    impulse_start = max(
        0,
        i - IMPULSE_LOOKBACK
    )

    impulse_low = float(
        df["Low"]
        .iloc[impulse_start:i + 1]
        .min()
    )

    if impulse_low <= 0:
        return False

    impulse_pct = (
        (a_price - impulse_low)
        / impulse_low
        * 100
    )

    if impulse_pct < MIN_IMPULSE_PCT:
        return False

    return True


# ============================================================
# 5. 扫描一只股票
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
            print(
                f"[SKIP] {symbol}: "
                f"Not enough data."
            )
            return None

        # 兼容yfinance新版MultiIndex
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in required_columns:

            if col not in df.columns:
                print(
                    f"[SKIP] {symbol}: "
                    f"Missing {col}."
                )
                return None

        df = df.dropna(
            subset=required_columns
        ).copy()

        if len(df) < 60:
            return None

        today_i = len(df) - 1

        today_high = float(
            df["High"].iloc[today_i]
        )

        today_low = float(
            df["Low"].iloc[today_i]
        )

        today_close = float(
            df["Close"].iloc[today_i]
        )

        today_date = df.index[today_i]

        # ====================================================
        # 寻找历史主波峰A
        # ====================================================

        earliest_a = max(
            PIVOT_LEFT,
            today_i - MAX_DAYS_AFTER_A
        )

        latest_a = (
            today_i -
            MIN_DAYS_AFTER_A
        )

        if latest_a <= earliest_a:
            return None

        candidate_indices = list(
            range(
                earliest_a,
                latest_a + 1
            )
        )

        # 优先检查距离今天最近的有效A
        candidate_indices.reverse()

        for a_i in candidate_indices:

            # --------------------------------------------
            # A首先必须是一个历史波峰
            # --------------------------------------------

            if not is_pivot_high(
                df["High"],
                a_i
            ):
                continue

            # --------------------------------------------
            # A还必须是一个相对明显的高点
            # --------------------------------------------

            if not is_major_high(
                df,
                a_i
            ):
                continue

            a_price = float(
                df["High"].iloc[a_i]
            )

            if a_price <= 0:
                continue

            a_date = df.index[a_i]

            # --------------------------------------------
            # A以后到昨天
            #
            # 今天不参与历史回撤的计算
            # --------------------------------------------

            after_a = df.iloc[
                a_i + 1:
                today_i
            ]

            if len(after_a) < 3:
                continue

            # --------------------------------------------
            # 找A以后最深回撤C
            # --------------------------------------------

            c_price = float(
                after_a["Low"].min()
            )

            c_date = (
                after_a["Low"]
                .idxmin()
            )

            pullback_pct = (
                (a_price - c_price)
                / a_price
                * 100
            )

            # --------------------------------------------
            # 中间回撤至少8%
            # --------------------------------------------

            if (
                pullback_pct
                < MIN_PULLBACK_PCT
            ):
                continue

            # --------------------------------------------
            # 定义重新冲击前高区域
            # --------------------------------------------

            lower_band = (
                a_price
                *
                (
                    1
                    -
                    RETEST_BELOW_PCT
                    / 100
                )
            )

            upper_band = (
                a_price
                *
                (
                    1
                    +
                    RETEST_ABOVE_PCT
                    / 100
                )
            )

            # --------------------------------------------
            # 中间必须真正离开A附近
            #
            # 至少有MIN_DAYS_AWAY天
            # 最高价都低于A区域下沿
            # --------------------------------------------

            away_days = int(
                (
                    after_a["High"]
                    < lower_band
                ).sum()
            )

            if away_days < MIN_DAYS_AWAY:
                continue

            # --------------------------------------------
            # 今天重新进入前高A附近
            #
            # 使用整根K是否与A区域相交
            #
            # 可以识别普通上涨和跳空上涨
            # --------------------------------------------

            today_enters_zone = (
                today_high >= lower_band
                and
                today_low <= upper_band
            )

            if not today_enters_zone:
                continue

            # --------------------------------------------
            # 昨天不能已经在这个区域
            #
            # 我们希望尽量在“首次重新进入”时提醒
            # --------------------------------------------

            yesterday_high = float(
                df["High"]
                .iloc[today_i - 1]
            )

            yesterday_low = float(
                df["Low"]
                .iloc[today_i - 1]
            )

            yesterday_in_zone = (
                yesterday_high
                >= lower_band
                and
                yesterday_low
                <= upper_band
            )

            if yesterday_in_zone:
                continue

            # --------------------------------------------
            # 如果昨天已经完全突破A区域上方，
            # 今天就不是从下方重新冲击
            # --------------------------------------------

            if (
                yesterday_high
                > upper_band
            ):
                continue

            # --------------------------------------------
            # 今天最高价距离A多少
            # --------------------------------------------

            distance_pct = (
                (today_high - a_price)
                / a_price
                * 100
            )

            bars_from_a = (
                today_i - a_i
            )

            print(
                f"[MATCH] {symbol} | "
                f"A={a_price:.2f} | "
                f"Pullback={pullback_pct:.2f}% | "
                f"TodayHigh={today_high:.2f} | "
                f"Distance={distance_pct:+.2f}%"
            )

            return {
                "symbol": symbol,

                "today_date":
                    str(
                        today_date.date()
                    ),

                "a_date":
                    str(
                        a_date.date()
                    ),

                "a_price":
                    a_price,

                "c_date":
                    str(
                        c_date.date()
                    ),

                "c_price":
                    c_price,

                "pullback_pct":
                    pullback_pct,

                "today_high":
                    today_high,

                "today_close":
                    today_close,

                "distance_pct":
                    distance_pct,

                "bars_from_a":
                    bars_from_a,

                "zone_low":
                    lower_band,

                "zone_high":
                    upper_band,

                "away_days":
                    away_days,
            }

        return None

    except Exception as exc:

        print(
            f"[ERROR] {symbol}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None


# ============================================================
# 6. 批量扫描 Nasdaq-100
# ============================================================

def run_scan():

    symbols = (
        get_nasdaq100_symbols()
    )

    print("")
    print(
        "================================"
    )

    print(
        f"Nasdaq-100 symbols: "
        f"{len(symbols)}"
    )

    print(
        "================================"
    )

    results = []

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        print(
            f"[{index}/{len(symbols)}] "
            f"Scanning {symbol}..."
        )

        result = scan_symbol(
            symbol
        )

        if result is not None:
            results.append(
                result
            )

    return results


# ============================================================
# 7. 生成邮件正文
# ============================================================

def build_email(results):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # ========================================================
    # 今天没有候选
    # ========================================================

    if not results:

        subject = (
            f"Nasdaq-100 前高再次冲击扫描 "
            f"{today}：无候选"
        )

        body = (
            f"{today}\n\n"
            "今天 Nasdaq-100 没有发现符合"
            "“明显回撤后再次冲击前高”"
            "价格结构的股票。\n\n"
            "当前主要筛选条件：\n"
            f"1. 中间回撤 >= "
            f"{MIN_PULLBACK_PCT:.1f}%\n"
            f"2. 前高区域下方容差 "
            f"{RETEST_BELOW_PCT:.1f}%\n"
            f"3. 前高区域上方容差 "
            f"{RETEST_ABOVE_PCT:.1f}%\n"
            f"4. 至少离开前高区域 "
            f"{MIN_DAYS_AWAY} 个交易日\n"
        )

        return subject, body

    # ========================================================
    # 有候选
    # ========================================================

    results = sorted(
        results,
        key=lambda x:
        abs(x["distance_pct"])
    )

    subject = (
        f"Nasdaq-100 前高再次冲击提醒 "
        f"{today}："
        f"{len(results)} 只"
    )

    lines = []

    lines.append(
        f"{today} Nasdaq-100 "
        f"前高再次冲击扫描结果"
    )

    lines.append("")

    lines.append(
        "以下股票在经历明显回撤后，"
        "今天重新进入历史主波峰附近。"
    )

    lines.append("")

    lines.append(
        "这只是价格结构初筛，"
        "需要你再打开图表人工判断。"
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
            f"最新交易日："
            f"{r['today_date']}"
        )

        lines.append("")

        lines.append(
            f"历史前高 A 日期："
            f"{r['a_date']}"
        )

        lines.append(
            f"历史前高 A："
            f"{r['a_price']:.2f}"
        )

        lines.append("")

        lines.append(
            f"中间最低点 C 日期："
            f"{r['c_date']}"
        )

        lines.append(
            f"中间最低点 C："
            f"{r['c_price']:.2f}"
        )

        lines.append(
            f"A → C 最大回撤："
            f"{r['pullback_pct']:.2f}%"
        )

        lines.append("")

        lines.append(
            f"今天最高价："
            f"{r['today_high']:.2f}"
        )

        lines.append(
            f"今天收盘价："
            f"{r['today_close']:.2f}"
        )

        lines.append(
            f"今天最高价相对前高A："
            f"{r['distance_pct']:+.2f}%"
        )

        lines.append("")

        lines.append(
            f"当前前高观察区域："
            f"{r['zone_low']:.2f}"
            f" ~ "
            f"{r['zone_high']:.2f}"
        )

        lines.append(
            f"A距今天："
            f"{r['bars_from_a']} 根日K"
        )

        lines.append(
            f"中间明显离开前高区域："
            f"{r['away_days']} 天"
        )

        lines.append("")

    return (
        subject,
        "\n".join(lines)
    )


# ============================================================
# 8. 发送QQ邮箱提醒
# ============================================================

def send_email(
    subject,
    body
):

    smtp_host = os.getenv(
        "SMTP_HOST"
    )

    smtp_port_text = os.getenv(
        "SMTP_PORT",
        "465"
    )

    smtp_user = os.getenv(
        "SMTP_USER"
    )

    smtp_pass = os.getenv(
        "SMTP_PASS"
    )

    email_to = os.getenv(
        "EMAIL_TO"
    )

    missing = []

    if not smtp_host:
        missing.append(
            "SMTP_HOST"
        )

    if not smtp_user:
        missing.append(
            "SMTP_USER"
        )

    if not smtp_pass:
        missing.append(
            "SMTP_PASS"
        )

    if not email_to:
        missing.append(
            "EMAIL_TO"
        )

    if missing:

        raise RuntimeError(
            "Missing GitHub Secrets: "
            +
            ", ".join(missing)
        )

    try:
        smtp_port = int(
            smtp_port_text
        )

    except ValueError:

        raise RuntimeError(
            "SMTP_PORT must be "
            "a number."
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

    print("")
    print(
        "Connecting to SMTP server..."
    )

    # QQ邮箱465端口
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

    # 如果以后改用587
    else:

        with smtplib.SMTP(
            smtp_host,
            smtp_port,
            timeout=30
        ) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(
                smtp_user,
                smtp_pass
            )

            server.sendmail(
                smtp_user,
                [email_to],
                msg.as_string()
            )

    print(
        "Email sent successfully."
    )


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Nasdaq-100 "
        "retest scanner..."
    )

    print("")

    results = run_scan()

    print("")
    print(
        "================================"
    )

    print(
        f"Found "
        f"{len(results)} "
        f"candidate(s)."
    )

    print(
        "================================"
    )

    if results:

        for r in results:

            print(
                r["symbol"],
                f"| A={r['a_price']:.2f}",
                f"| Pullback="
                f"{r['pullback_pct']:.2f}%",
                f"| TodayHigh="
                f"{r['today_high']:.2f}",
                f"| Distance="
                f"{r['distance_pct']:+.2f}%"
            )

    subject, body = (
        build_email(results)
    )

    send_email(
        subject,
        body
    )

    print("")
    print(
        "Scanner finished successfully."
    )
