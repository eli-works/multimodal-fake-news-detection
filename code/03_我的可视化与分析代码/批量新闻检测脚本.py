rows = parse_batch_file(file)                               # 校验文件格式与必填字段
batch_id = init_batch_state(total=len(rows))                # status=running, completed=0

for row in rows:
    task = build_task(row=row, batch_id=batch_id)
    send_to_worker(task)

for task in queue_consume(batch_id):
    try:
        features = prepare_multimodal_input(task)
        result = model_predict(features)                     # 检测模型推理
        save_single_result(task, result)                    # 回写单条结果
        update_batch_progress(batch_id, success=True)       # completed+1, success+1
    except Exception:
        save_single_failed(task)                            # 记录失败原因
        update_batch_progress(batch_id, success=False)      # completed+1, failed+1

finalize_batch_status(batch_id)




