import os
from celery_app import celery_engine
from core_agent import EnterpriseBrain

print("   ∟ [CELERY] Booting Resident ML Background Engine...")
celery_brain = EnterpriseBrain(
    db_path="/storage/enterprise_data_large/enterprise_erp_large.db",
    logs_path="/storage/vector_indices/project_logs"
)

@celery_engine.task(bind=True, name="tasks.execute_agent_audit")
def execute_agent_audit(self, question: str, thread_id: str):
    session_folder = os.path.join("/storage/user_uploads", thread_id)
    active_db = None
    
    if os.path.exists(session_folder):
        files = [f for f in os.listdir(session_folder) if f.endswith(('.db', '.csv'))]
        if files: active_db = os.path.join(session_folder, files[0])
        
    if not active_db:
        active_db = "/storage/enterprise_data_large/enterprise_erp_large.db"
        
    celery_brain.sql_worker.update_db_path(active_db)
    celery_brain.update_context(
        db_path=active_db, 
        logs_path="/storage/vector_indices/project_logs", 
        extra_folder=session_folder if os.path.exists(session_folder) else None
    )
    
    answer = celery_brain.run(question=question, thread_id=thread_id)
    return {"response": answer, "thread_id": thread_id}
