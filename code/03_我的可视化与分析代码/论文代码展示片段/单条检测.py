# 单条检测核心链路

# 网关：接收请求 -> 入库 -> 入队
async def detect_single(title, content, image_paths, user_id):
    task_id = save_detection(user_id, title, content, image_paths, status="pending")
    set_task_state(task_id, status="pending")
    celery_send("run_single_detection", task_id, title, content, image_paths)
    return {"task_id": task_id}


# Worker：异步消费任务 -> 调用模型服务 -> 回写结果
def run_single_detection(task_id, title, content, image_paths):
    set_task_state(task_id, status="running")
    label, confidence = model_service_predict(title, content, image_paths)
    save_result(task_id, label, confidence)  # 写数据库
    set_task_state(task_id, status="success", prediction=label, confidence=confidence)



# 模型服务：预处理 -> 推理 -> 解码输出
def model_service_predict(title, content, image_paths):
    text_ids, text_mask = text_preprocess(title, content)
    image_tensor = image_preprocess(image_paths)
    with torch.no_grad():
        score = model(text_ids, text_mask, image_tensor)
    return decode_label(score), decode_confidence(score)
