from celery import Celery

# Attaches directly to your existing internal Redis container for $0.00
celery_engine = Celery(
    "enterprise_ml_engine",
    broker="redis://redis-broker:6379/0",
    backend="redis://redis-broker:6379/1"
)

celery_engine.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # 🔒 CRITICAL WIN: Explicitly lock worker concurrency to 1.
    # This dedicates 100% of physical CPU core cache to one heavy ML reranking pass at a time!
    worker_concurrency=1,
    task_track_started=True
)

import tasks
