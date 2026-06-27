import json
import time
import uuid
import random
from locust import HttpUser, task, between, events

# A pool of authentic corporate audit prompts to prevent artificial L1 cache 100% hit rates
AUDIT_PROMPTS = [
    "Calculate peak 24h transaction velocity for users with credit under 600.",
    "What is the total financial volume of transactions flagged with VPN usage?",
    "Summarize the fraud distribution across different device operating systems.",
    "Identify the maximum transaction velocity for high-risk corporate accounts.",
    "What is the average transaction velocity for fraudulent records?"
]

class EnterpriseAuditUser(HttpUser):
    # Simulates a natural human "think time" delay between 2 to 5 seconds between actions
    wait_time = between(2.0, 5.0)

    def on_start(self):
        """Initializes a unique corporate session thread upon spawning."""
        self.thread_id = str(uuid.uuid4())
        self.headers = {
            "Content-Type": "application/json",
            # 🎯 Corrected Let's Encrypt Basic Auth passkey for "guest:hireme2026"
            "Authorization": "Basic Z3Vlc3Q6aGlyZW1lMjAyNg=="
        }

    @task
    def execute_asynchronous_audit(self):
        prompt = random.choice(AUDIT_PROMPTS)
        payload = {
            "question": prompt,
            "thread_id": self.thread_id,
            "clearance_level": "tier_1"
        }

        # 1. Hit the non-blocking ingestion endpoint
        with self.client.post("/api/v1/audit/async", data=json.dumps(payload), headers=self.headers, catch_response=True) as response:
            if response.status_code not in [200, 202]:
                response.failure(f"Ingestion rejected: HTTP {response.status_code}")
                return

            data = response.json()
            # If L1 cache caught it instantly, finish task
            if data.get("cached"):
                response.success()
                return

            task_id = data.get("task_id")
            if not task_id:
                response.failure("API failed to return a tracking task_id.")
                return

        # 2. Authentic Polling Loop: Check task status every 3 seconds until finished
        max_polling_attempts = 40  # 2 minutes timeout ceiling
        attempts = 0
        
        while attempts < max_polling_attempts:
            time.sleep(3.0)
            attempts += 1
            
            with self.client.get(f"/api/v1/audit/status/{task_id}", headers=self.headers, name="/api/v1/audit/status/[id]", catch_response=True) as poll_res:
                if poll_res.status_code != 200:
                    poll_res.failure(f"Polling endpoint failed: HTTP {poll_res.status_code}")
                    break

                status_data = poll_res.json()
                current_status = status_data.get("status")

                if current_status == "SUCCESS":
                    poll_res.success()
                    return
                elif current_status == "FAILURE":
                    poll_res.failure(f"Celery ML background task crashed: {status_data.get('error')}")
                    return
                # If PENDING or STARTED, loop and poll again