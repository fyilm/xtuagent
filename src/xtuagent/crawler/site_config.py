"""目标站点清单。

⚠️ 域名会随学校改版迁移，**同一学院往往有新旧两套域名**，而且新旧域名
的 WAF 放行策略并不一致：实测（2026-09-30）旧域名 `cie/bs/chem/mse/mee/cee/
fl/lsxy/gggl/marx/phy/xgb/zhaosheng` 对非校园网出口一律返回 403，而学校现行
官网用的是另一套域名（`jwxy/business/hxxy/clxy/wgyxy/wlxy/hgxy/cem/glxy/
jxgc/mks/xtuxgb/zs`），同一时刻访问**返回 200 正常页面**。

所以：**不要靠猜域名，去研究生院/招生网的「各学院方案链接汇总」页拿现行域名**，
例如 https://yjsc.xtu.edu.cn/info/1084/10250.htm 就逐条列出了全部学院网址。

下面把两套域名都保留：新域名在当前网络可用，旧域名留给校园网出口的用户
（失败只花一个请求，代价可忽略）。
"""

# 校级机关 / 职能部门
_OFFICES = [
    {"name": "湘潭大学官网", "url": "https://www.xtu.edu.cn"},
    {"name": "教务处", "url": "https://jwc.xtu.edu.cn"},
    {"name": "研究生院", "url": "https://yjsc.xtu.edu.cn"},
    {"name": "招生网", "url": "https://zhaosheng.xtu.edu.cn"},  # 旧域名，多数网络 403
    {"name": "本科招生网", "url": "https://zs.xtu.edu.cn"},  # 现行域名，可达 ★
    {"name": "学生工作部", "url": "https://xgb.xtu.edu.cn"},  # 旧域名，多数网络 403
    {"name": "学生工作部（现行）", "url": "https://xtuxgb.xtu.edu.cn"},  # ★
    {"name": "国际交流处", "url": "https://gjjl.xtu.edu.cn"},
    {"name": "网络与信息中心", "url": "https://nic.xtu.edu.cn"},
    {"name": "就业指导中心", "url": "https://job.xtu.edu.cn"},
    {"name": "人事处", "url": "https://rsc.xtu.edu.cn"},
    {"name": "社科处", "url": "https://skc.xtu.edu.cn"},
    {"name": "科技处", "url": "https://kjc.xtu.edu.cn"},
    {"name": "计划财务处", "url": "https://cwc.xtu.edu.cn"},
    {"name": "图书馆", "url": "https://lib.xtu.edu.cn"},  # JS 骨架页，需 --render
]

# 学院 / 学部（现行域名，2026-09-30 实测可达 ★）
_COLLEGES_CURRENT = [
    {"name": "计算机学院·网络空间安全学院", "url": "https://jwxy.xtu.edu.cn"},
    {"name": "商学院", "url": "https://business.xtu.edu.cn"},
    {"name": "哲学与历史文化学院（碧泉书院）", "url": "https://bqsy.xtu.edu.cn"},
    {"name": "化学学院", "url": "https://hxxy.xtu.edu.cn"},
    {"name": "材料科学与工程学院", "url": "https://clxy.xtu.edu.cn"},
    {"name": "外国语学院", "url": "https://wgyxy.xtu.edu.cn"},
    {"name": "物理与光电工程学院", "url": "https://wlxy.xtu.edu.cn"},
    {"name": "化工学院", "url": "https://hgxy.xtu.edu.cn"},
    {"name": "土木工程学院", "url": "https://cem.xtu.edu.cn"},
    {"name": "公共管理学院", "url": "https://glxy.xtu.edu.cn"},
    {"name": "机械工程与力学学院", "url": "https://jxgc.xtu.edu.cn"},
    {"name": "马克思主义学院", "url": "https://mks.xtu.edu.cn"},
    {"name": "法学学部", "url": "https://fxxb.xtu.edu.cn"},
    {"name": "兴湘学院", "url": "https://xxxy.xtu.edu.cn"},
]

# 学院 / 学部（沿用的旧域名：非校园网出口普遍 403，校园网内可能可达）
_COLLEGES_LEGACY = [
    {"name": "数学与计算科学学院", "url": "https://math.xtu.edu.cn"},
    {"name": "物理与光电工程学院（旧）", "url": "https://phy.xtu.edu.cn"},
    {"name": "化学学院（旧）", "url": "https://chem.xtu.edu.cn"},
    {"name": "材料科学与工程学院（旧）", "url": "https://mse.xtu.edu.cn"},
    {"name": "计算机学院（旧）", "url": "https://cie.xtu.edu.cn"},
    {"name": "机械工程与力学学院（旧）", "url": "https://mee.xtu.edu.cn"},
    {"name": "土木工程学院（旧）", "url": "https://cee.xtu.edu.cn"},
    {"name": "环境与资源学院", "url": "https://hjzy.xtu.edu.cn"},
    {"name": "自动化与电子信息学院", "url": "https://aei.xtu.edu.cn"},
    {"name": "文学与新闻学院", "url": "https://wxy.xtu.edu.cn"},
    {"name": "外国语学院（旧）", "url": "https://fl.xtu.edu.cn"},
    {"name": "历史文化学院（旧）", "url": "https://lsxy.xtu.edu.cn"},
    {"name": "商学院（旧）", "url": "https://bs.xtu.edu.cn"},
    {"name": "法学学部（旧）", "url": "https://law.xtu.edu.cn"},
    {"name": "公共管理学院（旧）", "url": "https://gggl.xtu.edu.cn"},
    {"name": "马克思主义学院（旧）", "url": "https://marx.xtu.edu.cn"},
    {"name": "艺术学院", "url": "https://art.xtu.edu.cn"},
]

TARGET_SITES = _OFFICES + _COLLEGES_CURRENT + _COLLEGES_LEGACY

CRAWL_RULES = {
    "allowed_domains": ["xtu.edu.cn"],
    # 优先级规则：命中这些栏目特征的链接优先入队，
    # 避免在页数上限内被大量低价值页面（首页栏目、导航页）挤占额度。
    "priority_patterns": [
        r"/info/\d+/\d+\.htm",
        r"/tzgg/",
        r"/xwzx/",
        r"/jxgz/",
        r"/xsgz/",
        r"/pyfa/",
        r"/gzzd/",
        r"/jxky/",
        r"/bksjy/",
        r"/yjsjy/",
    ],
    # 正文页判定：仅用于日志标注（`*` 前缀），不影响抓取范围。
    "content_patterns": [
        r"/info/\d+/\d+\.htm",
        r"/content/",
        r"/article/",
        r"\d+\.htm$",
        r"\d+\.html$",
    ],
    # 排除规则：命中即不入队、不抓取。
    "exclude_patterns": [
        r"\.jpg$",
        r"\.png$",
        r"\.gif$",
        r"\.mp4$",
        r"\.avi$",
        r"/video/",
        r"/images/",
        r"/img/",
        r"javascript:",
        r"mailto:",
    ],
    # 可下载的文档类型。注意：.doc/.docx 目前无法解析为文本
    # （TextExtractor 明确跳过），因此不再纳入，避免下载后无产出。
    "file_types": [".pdf", ".md", ".txt"],
}
