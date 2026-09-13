import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.header import Header

import pandas as pd
import yfinance as yf


# ============================================================
# 1. 当前 Nasdaq-100 股票池
#
# Nasdaq-100 是100家公司。
# 因部分公司存在多个股票类别，因此证券代码可能超过100个。
#
# 后续指数换股时，只需要更新这里。
# ============================================================

NASDAQ100_SYMBOLS = [
    "AAPL",
    "APP",
    "TTWO",
    "ADSK",
    "CMCSA",
    "PYPL",
    "ISRG",
    "DASH",
    "MELI",
    "HONA",
    "NXPI",
    "CSX",
    "FANG",
    "WBD",
    "FTNT",
    "PANW",
    "SNPS",
    "ADP",
    "DXCM",
    "GOOG",
    "GOOGL",
    "MNST",
    "PCAR",
    "BKNG",
    "CRWD",
    "FAST",
    "SPCX",
    "PAYX",
    "QCOM",
    "MSFT",
    "CDNS",
    "ROST",
    "MDLZ",
    "COST",
    "NFLX",
    "PEP",
    "WMT",
    "TMUS",
    "SHOP",
    "AMZN",
    "INTU",
    "ROP",
    "WDAY",
    "GILD",
    "ORLY",
    "EXC",
    "MAR",
    "ODFL",
    "CTAS",
    "SBUX",
    "KHC",
    "CCEP",
    "AEP",
    "AVGO",
    "PDD",
    "XEL",
    "ADI",
    "TXN",
    "LIN",
    "TSLA",
    "ABNB",
    "VRTX",
    "TRI",
    "HON",
    "META",
    "GEHC",
    "MPWR",
    "DDOG",
    "IDXX",
    "RKLB",
    "REGN",
    "CSCO",
    "FER",
    "KDP",
    "MCHP",
    "PLTR",
    "AMGN",
    "AXON",
    "ADBE",
    "NVDA",
    "ASML",
    "STX",
    "CEG",
    "MSTR",
    "KLAC",
    "AMAT",
    "AMD",
    "TER",
    "ALNY",
    "MRVL",
    "ARM",
    "CPRT",
    "SNDK",
    "WDC",
    "MU",
    "NBIS",
    "ALAB",
    "LITE",
    "INTC",
    "LRCX",
    "CRWV",
    "BKR",
]


# ============================================================
# 2. 可调参数
# ============================================================

# ------------------------------------------------------------
# 历史前高 A 的初步识别
# ------------------------------------------------------------

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# A必须在最近多少根K中属于明显高位
A_CONTEXT_BARS = 30

# A允许距离这段时间最高价多少
# 2 = A只要在近期最高价2%以内
A_EXTREME_TOLERANCE_PCT = 2.0


# ------------------------------------------------------------
# A之前需要有一定上涨推动
# ------------------------------------------------------------

IMPULSE_LOOKBACK = 15

MIN_IMPULSE_PCT = 4.0


# ------------------------------------------------------------
# A -> C 回撤
#
# 这是目前很重要的参数。
# 默认要求从A至少回撤8%。
# ------------------------------------------------------------

MIN_PULLBACK_PCT = 8.0


# ------------------------------------------------------------
# “真正离开前高”
#
# A锁定不仅要求跌8%，
# 还要求至少有若干天的最高价低于前高区域下沿。
#
# 这样尽量排除高位横盘。
# ------------------------------------------------------------

MIN_DAYS_AWAY = 3


# ------------------------------------------------------------
# “再次冲击前高”的价格区域
#
# 例如：
#
# A = 100
#
# 下方6% = 94
# 上方8% = 108
#
# 当前K线重新进入94~108区域时，
# 就认为正在重新攻击历史前高。
#
# 这是目前为了保证正例不漏掉而设置得比较宽。
# ------------------------------------------------------------

RETEST_BELOW_PCT = 6.0

RETEST_ABOVE_PCT = 8.0


# ------------------------------------------------------------
# A 与今天之间的时间
# ------------------------------------------------------------

MIN_DAYS_AFTER_A = 5

MAX_DAYS_AFTER_A = 120


# ------------------------------------------------------------
# 下载历史数据
# ------------------------------------------------------------

DOWNLOAD_PERIOD = "1y"


# ============================================================
# 3. 判断某根历史K是不是初步波峰
# ============================================================

def is_pivot_high(df, i):

    if i < PIVOT_LEFT:
        return False

    if i + PIVOT_RIGHT >= len(df):
        return False

    center = float(
        df["High"].iloc[i]
    )

    left = df["High"].iloc[
        i - PIVOT_LEFT:
        i
    ]

    right = df["High"].iloc[
        i + 1:
        i + PIVOT_RIGHT + 1
    ]

    if len(left) < PIVOT_LEFT:
        return False

    if len(right) < PIVOT_RIGHT:
        return False

    return (
        center >= float(left.max())
        and
        center > float(right.max())
    )


# ============================================================
# 4. 判断这个波峰够不够明显
# ============================================================

def is_major_high(df, i):

    a_price = float(
        df["High"].iloc[i]
    )

    # --------------------------------------------------------
    # 近期高点
    # --------------------------------------------------------

    context_start = max(
        0,
        i - A_CONTEXT_BARS + 1
    )

    context_high = float(
        df["High"]
        .iloc[context_start:i + 1]
        .max()
    )

    minimum_allowed = (
        context_high
        *
        (
            1
            -
            A_EXTREME_TOLERANCE_PCT
            / 100
        )
    )

    if a_price < minimum_allowed:
        return False

    # --------------------------------------------------------
    # A之前要有一定上涨推动
    # --------------------------------------------------------

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
        /
        impulse_low
        *
        100
    )

    if impulse_pct < MIN_IMPULSE_PCT:
        return False

    return True


# ============================================================
# 5. 清洗 yfinance 数据
# ============================================================

def clean_dataframe(df):

    if df is None:
        return None

    if len(df) == 0:
        return None

    # --------------------------------------------------------
    # yfinance 某些情况下返回 MultiIndex
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    needed = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for col in needed:

        if col not in df.columns:
            return None

    df = (
        df[needed]
        .copy()
        .dropna()
    )

    if len(df) < 60:
        return None

    return df


# ============================================================
# 6. 扫描一只股票
#
# 核心状态：
#
# 状态0：
#   没有A
#
# 状态1：
#   找到候选A
#   A尚未锁定
#
#   此阶段继续创新高 -> A更新
#
# 状态2：
#   A已经经历 >=8% 回撤
#   并且价格真正离开顶部区域
#
#   A正式锁定
#
#   此后任何较低的小波峰都不能替换A
#
#   等待第一次重新攻击A
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

        df = clean_dataframe(df)

        if df is None:

            print(
                f"[SKIP] {symbol}: "
                f"no usable data"
            )

            return None

        today_i = len(df) - 1

        if today_i < 50:
            return None

        today_high = float(
            df["High"].iloc[today_i]
        )

        today_low = float(
            df["Low"].iloc[today_i]
        )

        today_close = float(
            df["Close"].iloc[today_i]
        )

        today_date = (
            df.index[today_i]
        )

        # ----------------------------------------------------
        # 为了避免太古老的结构，
        # 只从一定范围内开始寻找。
        # ----------------------------------------------------

        start_i = max(
            PIVOT_LEFT + PIVOT_RIGHT,
            today_i
            -
            MAX_DAYS_AFTER_A
            -
            A_CONTEXT_BARS
        )

        # ----------------------------------------------------
        # 状态变量
        # ----------------------------------------------------

        state = 0

        a_i = None
        a_price = None
        a_date = None

        lowest_since_a = None
        c_i = None

        away_days = 0

        locked_pullback_pct = None

        locked_c_price = None
        locked_c_date = None

        # ====================================================
        # 一天一天模拟到“昨天”
        #
        # 今天不能用于建立历史结构。
        # ====================================================

        for j in range(
            start_i,
            today_i
        ):

            # ------------------------------------------------
            # 已有结构太久 -> 放弃
            # ------------------------------------------------

            if (
                a_i is not None
                and
                j - a_i
                > MAX_DAYS_AFTER_A
            ):

                state = 0

                a_i = None
                a_price = None
                a_date = None

                lowest_since_a = None
                c_i = None

                away_days = 0

                locked_pullback_pct = None
                locked_c_price = None
                locked_c_date = None

            # =================================================
            # 状态0：
            # 没有A
            # =================================================

            if state == 0:

                # pivot 要到右边几根K出现后才能确认
                pivot_i = (
                    j - PIVOT_RIGHT
                )

                if (
                    pivot_i
                    >= start_i
                    and
                    is_pivot_high(
                        df,
                        pivot_i
                    )
                    and
                    is_major_high(
                        df,
                        pivot_i
                    )
                ):

                    state = 1

                    a_i = pivot_i

                    a_price = float(
                        df["High"]
                        .iloc[pivot_i]
                    )

                    a_date = (
                        df.index[pivot_i]
                    )

                    lowest_since_a = (
                        a_price
                    )

                    c_i = pivot_i

                    away_days = 0

                continue

            # =================================================
            # 状态1：
            # 有候选A，但是还没有真正形成大回撤
            #
            # 这时如果继续创新高：
            #
            # A必须更新。
            #
            # 这就是之前 AFL 等问题需要修正的地方。
            # =================================================

            if state == 1:

                current_high = float(
                    df["High"].iloc[j]
                )

                current_low = float(
                    df["Low"].iloc[j]
                )

                # ------------------------------------------------
                # C确认以前继续创新高：
                # A更新为最新最高价。
                # ------------------------------------------------

                if current_high > a_price:

                    a_i = j

                    a_price = (
                        current_high
                    )

                    a_date = (
                        df.index[j]
                    )

                    lowest_since_a = (
                        current_low
                    )

                    c_i = j

                    away_days = 0

                    continue

                # ------------------------------------------------
                # 更新A后的最低价
                # ------------------------------------------------

                if (
                    lowest_since_a
                    is None
                    or
                    current_low
                    <
                    lowest_since_a
                ):

                    lowest_since_a = (
                        current_low
                    )

                    c_i = j

                # ------------------------------------------------
                # 当前最大回撤
                # ------------------------------------------------

                pullback_pct = (
                    (
                        a_price
                        -
                        lowest_since_a
                    )
                    /
                    a_price
                    *
                    100
                )

                # ------------------------------------------------
                # 前高区域下沿
                # ------------------------------------------------

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

                # ------------------------------------------------
                # 有多少天真正离开A附近
                # ------------------------------------------------

                if (
                    current_high
                    <
                    lower_band
                ):

                    away_days += 1

                # ------------------------------------------------
                # 满足两个条件后：
                #
                # 1. 回撤 >= 8%
                # 2. 至少离开顶部区域3天
                #
                # A正式锁定。
                # ------------------------------------------------

                if (
                    pullback_pct
                    >=
                    MIN_PULLBACK_PCT
                    and
                    away_days
                    >=
                    MIN_DAYS_AWAY
                ):

                    state = 2

                    locked_pullback_pct = (
                        pullback_pct
                    )

                    locked_c_price = (
                        lowest_since_a
                    )

                    locked_c_date = (
                        df.index[c_i]
                    )

                continue

            # =================================================
            # 状态2：
            #
            # A已经锁定。
            #
            # 后面的小波峰不能再改变A。
            #
            # 如果历史上已经重新进入A附近，
            # 说明这次机会早就发生过，
            # 当前这轮结构结束。
            # =================================================

            if state == 2:

                current_high = float(
                    df["High"].iloc[j]
                )

                current_low = float(
                    df["Low"].iloc[j]
                )

                # ------------------------------------------------
                # C在真正重新冲击A以前仍允许继续向下延伸
                # ------------------------------------------------

                if (
                    current_low
                    <
                    locked_c_price
                ):

                    locked_c_price = (
                        current_low
                    )

                    locked_c_date = (
                        df.index[j]
                    )

                    locked_pullback_pct = (
                        (
                            a_price
                            -
                            locked_c_price
                        )
                        /
                        a_price
                        *
                        100
                    )

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

                # ------------------------------------------------
                # 历史上已经重新进入过前高区域
                #
                # 那么这套A已经被“测试”过，
                # 今天不能再次拿它发第一次提醒。
                # ------------------------------------------------

                entered_zone = (
                    current_high
                    >=
                    lower_band
                    and
                    current_low
                    <=
                    upper_band
                )

                if entered_zone:

                    state = 0

                    a_i = None
                    a_price = None
                    a_date = None

                    lowest_since_a = None
                    c_i = None

                    away_days = 0

                    locked_pullback_pct = None
                    locked_c_price = None
                    locked_c_date = None

                continue

        # ====================================================
        # 历史模拟完成。
        #
        # 到昨天为止必须仍然存在一个：
        #
        # “已经锁定，但尚未重新测试”的A。
        # ====================================================

        if state != 2:
            return None

        if a_i is None:
            return None

        bars_from_a = (
            today_i - a_i
        )

        if (
            bars_from_a
            <
            MIN_DAYS_AFTER_A
        ):
            return None

        # ----------------------------------------------------
        # 今天的A附近区域
        # ----------------------------------------------------

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

        # ====================================================
        # 最重要的一步：
        #
        # 今天是否第一次重新进入历史前高区域？
        #
        # 用当天整根K线与区域是否相交判断。
        #
        # 因此普通上涨和跳空上涨都可以抓到。
        # ====================================================

        today_enters_zone = (
            today_high
            >=
            lower_band
            and
            today_low
            <=
            upper_band
        )

        if not today_enters_zone:
            return None

        # ----------------------------------------------------
        # 今天最高价距离A多少
        # ----------------------------------------------------

        distance_pct = (
            (
                today_high
                -
                a_price
            )
            /
            a_price
            *
            100
        )

        print(
            f"[MATCH] {symbol} | "
            f"A={a_price:.2f} | "
            f"Pullback="
            f"{locked_pullback_pct:.2f}% | "
            f"TodayHigh="
            f"{today_high:.2f} | "
            f"Distance="
            f"{distance_pct:+.2f}%"
        )

        return {
            "symbol":
                symbol,

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
                    locked_c_date.date()
                ),

            "c_price":
                locked_c_price,

            "pullback_pct":
                locked_pullback_pct,

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

    except Exception as exc:

        print(
            f"[ERROR] {symbol}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None


# ============================================================
# 7. 批量扫描 Nasdaq-100
# ============================================================

def run_scan():

    symbols = (
        NASDAQ100_SYMBOLS
    )

    print("")
    print(
        "================================"
    )

    print(
        "Nasdaq-100 retest scanner"
    )

    print(
        f"Symbols to scan: "
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
# 8. 生成邮件
# ============================================================

def build_email(results):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # 没有候选
    # --------------------------------------------------------

    if not results:

        subject = (
            "Nasdaq-100 前高再次冲击扫描 "
            f"{today}：无候选"
        )

        body = (
            f"{today}\n\n"

            "今天 Nasdaq-100 没有发现符合条件的"
            "“明显回撤后再次冲击历史前高”股票。\n\n"

            "当前条件：\n"

            f"历史前高后至少回撤："
            f"{MIN_PULLBACK_PCT:.1f}%\n"

            f"至少明显离开前高区域："
            f"{MIN_DAYS_AWAY} 个交易日\n"

            f"再次冲击区域："
            f"A下方 {RETEST_BELOW_PCT:.1f}% "
            f"至 A上方 {RETEST_ABOVE_PCT:.1f}%\n\n"

            "没有候选也会发送本邮件，"
            "用于确认自动扫描器每天正常运行。"
        )

        return (
            subject,
            body
        )

    # --------------------------------------------------------
    # 有候选
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x:
        abs(
            x["distance_pct"]
        )
    )

    subject = (
        "Nasdaq-100 前高再次冲击提醒 "
        f"{today}："
        f"{len(results)} 只"
    )

    lines = []

    lines.append(
        f"{today} Nasdaq-100 "
        f"扫描结果"
    )

    lines.append("")

    lines.append(
        "以下股票在经历明显回撤以后，"
        "今天第一次重新进入历史前高附近。"
    )

    lines.append("")

    lines.append(
        "请打开日K图进行人工判断。"
    )

    lines.append("")

    lines.append(
        "本邮件只是价格结构筛选，"
        "不是交易建议。"
    )

    lines.append("")

    for r in results:

        lines.append(
            "================================"
        )

        lines.append(
            f"股票："
            f"{r['symbol']}"
        )

        lines.append("")

        lines.append(
            f"最新交易日："
            f"{r['today_date']}"
        )

        lines.append(
            f"历史前高A日期："
            f"{r['a_date']}"
        )

        lines.append(
            f"历史前高A："
            f"{r['a_price']:.2f}"
        )

        lines.append("")

        lines.append(
            f"中间最低点C日期："
            f"{r['c_date']}"
        )

        lines.append(
            f"中间最低点C："
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
            f"今天最高价相对A："
            f"{r['distance_pct']:+.2f}%"
        )

        lines.append("")

        lines.append(
            f"当前前高观察区："
            f"{r['zone_low']:.2f}"
            " ~ "
            f"{r['zone_high']:.2f}"
        )

        lines.append(
            f"A距今天："
            f"{r['bars_from_a']} "
            f"根日K"
        )

        lines.append("")

    return (
        subject,
        "\n".join(lines)
    )


# ============================================================
# 9. QQ邮箱发送
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

    smtp_port = int(
        smtp_port_text
    )

    message = MIMEText(
        body,
        "plain",
        "utf-8"
    )

    message["Subject"] = Header(
        subject,
        "utf-8"
    )

    message["From"] = (
        smtp_user
    )

    message["To"] = (
        email_to
    )

    print("")
    print(
        "Connecting to email server..."
    )

    # --------------------------------------------------------
    # QQ邮箱465端口
    # --------------------------------------------------------

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
                message.as_string()
            )

    # --------------------------------------------------------
    # 兼容587端口
    # --------------------------------------------------------

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
                message.as_string()
            )

    print(
        "Email sent successfully."
    )


# ============================================================
# 10. 主程序
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Nasdaq-100 "
        "retest scanner..."
    )

    results = (
        run_scan()
    )

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
                f"| A="
                f"{r['a_price']:.2f}",
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
