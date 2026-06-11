


@celery_app.task(bind=True, max_retries=3, name="worker.tasks.run_single_detection")
def run_single_detection(self, x):
    try:
        return run_async(_run_detection(...))
    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        run_async(_persist_failure(detection_id, task_id, str(exc)))
        raise