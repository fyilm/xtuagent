TARGET_SITES = [
    {"name": "湘潭大学官网", "url": "https://www.xtu.edu.cn"},
    {"name": "教务处", "url": "https://jwc.xtu.edu.cn"},
    {"name": "研究生院", "url": "https://yjsc.xtu.edu.cn"},
    {"name": "学生工作部", "url": "https://xgb.xtu.edu.cn"},
    {"name": "国际交流处", "url": "https://gjjl.xtu.edu.cn"},
    {"name": "招生网", "url": "https://zhaosheng.xtu.edu.cn"},
    {"name": "图书馆", "url": "https://lib.xtu.edu.cn"},
    {"name": "网络与信息中心", "url": "https://nic.xtu.edu.cn"},
    {"name": "就业指导中心", "url": "https://job.xtu.edu.cn"},
    {"name": "数学与计算科学学院", "url": "https://math.xtu.edu.cn"},
    {"name": "物理与光电工程学院", "url": "https://phy.xtu.edu.cn"},
    {"name": "化学学院", "url": "https://chem.xtu.edu.cn"},
    {"name": "材料科学与工程学院", "url": "https://mse.xtu.edu.cn"},
    {"name": "计算机学院", "url": "https://cie.xtu.edu.cn"},
    {"name": "机械工程与力学学院", "url": "https://mee.xtu.edu.cn"},
    {"name": "土木工程学院", "url": "https://cee.xtu.edu.cn"},
    {"name": "环境与资源学院", "url": "https://hjzy.xtu.edu.cn"},
    {"name": "自动化与电子信息学院", "url": "https://aei.xtu.edu.cn"},
    {"name": "文学与新闻学院", "url": "https://wxy.xtu.edu.cn"},
    {"name": "外国语学院", "url": "https://fl.xtu.edu.cn"},
    {"name": "历史文化学院", "url": "https://lsxy.xtu.edu.cn"},
    {"name": "商学院", "url": "https://bs.xtu.edu.cn"},
    {"name": "法学学部", "url": "https://law.xtu.edu.cn"},
    {"name": "公共管理学院", "url": "https://gggl.xtu.edu.cn"},
    {"name": "马克思主义学院", "url": "https://marx.xtu.edu.cn"},
    {"name": "艺术学院", "url": "https://art.xtu.edu.cn"},
]

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
