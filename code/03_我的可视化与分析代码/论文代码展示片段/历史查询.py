

def get_history_list(user_id, page, page_size, keyword, start_time, end_time):
    # 仅查询当前登录用户数据
    filters = ["user_id匹配当前登录用户"]
    # 关键词与时间范围筛选
    if keyword:
        filters += ["标题或正文包含keyword"]
    if start_time:
        filters += ["created_at >= start_time"]
    if end_time:
        filters += ["created_at <= end_time"]
    # 先查总数再分页
    total = "查询符合条件的总记录数"
    rows = "按created_at倒序分页查询(page, page_size, filters)"
    return total, rows

