from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

@dataclass
class AutomatedTask:
    task_type: str
    due_date: datetime
    assigned_to: str
    description: str
    priority: str
    status: str = "pending"

class SalesAutomation:
    def __init__(self):
        self.email_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "your_email@example.com",
            "password": "your_password"
        }
        
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )

    def get_tasks(self, assigned_to: Optional[str] = None, 
                 priority: Optional[str] = None, 
                 status: Optional[str] = None) -> List[Dict]:
        """Get filtered tasks based on criteria from Chroma database"""
        # Construct the query based on filters
        filter_parts = []
        if assigned_to:
            filter_parts.append(f"assigned_to: {assigned_to}")
        if priority:
            filter_parts.append(f"priority: {priority}")
        if status:
            filter_parts.append(f"status: {status}")
            
        query = " AND ".join(filter_parts) if filter_parts else "type: task"
        
        # Query the vector store
        results = self.vector_store.similarity_search(
            query,
            k=100  # Adjust based on your needs
        )
        
        # Process results into task dictionaries
        tasks = []
        for doc in results:
            if hasattr(doc, 'metadata') and 'task_data' in doc.metadata:
                try:
                    task_data = json.loads(doc.metadata['task_data'])
                    # Apply filters manually to ensure accuracy
                    if self._matches_filters(task_data, assigned_to, priority, status):
                        tasks.append(task_data)
                except json.JSONDecodeError:
                    continue
                    
        return tasks

    def _matches_filters(self, task: Dict, assigned_to: Optional[str], 
                        priority: Optional[str], status: Optional[str]) -> bool:
        """Helper method to check if a task matches the given filters"""
        if assigned_to and task.get('assigned_to') != assigned_to:
            return False
        if priority and task.get('priority') != priority:
            return False
        if status and task.get('status') != status:
            return False
        return True

    def update_task(self, task_id: str, task_data: Dict) -> Dict:
        """Update task in Chroma database"""
        # Create the task document
        task_doc = f"""
        Task ID: {task_id}
        Type: {task_data.get('task_type', 'unknown')}
        Assigned To: {task_data.get('assigned_to', 'unassigned')}
        Priority: {task_data.get('priority', 'medium')}
        Status: {task_data.get('status', 'pending')}
        Description: {task_data.get('description', '')}
        """
        
        # Update metadata
        metadata = {
            'task_id': task_id,
            'task_data': json.dumps({**task_data, 'id': task_id, 'updated_at': datetime.now().isoformat()})
        }
        
        # Add to vector store
        self.vector_store.add_texts(
            texts=[task_doc],
            metadatas=[metadata]
        )
        
        return {**task_data, 'id': task_id, 'updated_at': datetime.now().isoformat()}

    def get_follow_ups(self, client: Optional[str] = None,
                      date_from: Optional[str] = None,
                      date_to: Optional[str] = None) -> List[Dict]:
        """Get follow-ups from Chroma database"""
        query_parts = ["type: follow_up"]
        if client:
            query_parts.append(f"client: {client}")
            
        query = " AND ".join(query_parts)
        
        results = self.vector_store.similarity_search(
            query,
            k=100
        )
        
        follow_ups = []
        for doc in results:
            if hasattr(doc, 'metadata') and 'follow_up_data' in doc.metadata:
                try:
                    follow_up = json.loads(doc.metadata['follow_up_data'])
                    # Apply date filters
                    if self._matches_date_range(follow_up.get('date'), date_from, date_to):
                        follow_ups.append(follow_up)
                except json.JSONDecodeError:
                    continue
                    
        return follow_ups

    def _matches_date_range(self, date_str: Optional[str], 
                          date_from: Optional[str], 
                          date_to: Optional[str]) -> bool:
        """Helper method to check if a date falls within the given range"""
        if not date_str:
            return False
            
        try:
            date = datetime.fromisoformat(date_str)
            if date_from and date < datetime.fromisoformat(date_from):
                return False
            if date_to and date > datetime.fromisoformat(date_to):
                return False
            return True
        except ValueError:
            return False

    def get_alerts(self, alert_type: Optional[str] = None,
                  priority: Optional[str] = None,
                  date_from: Optional[str] = None) -> List[Dict]:
        """Get alerts from Chroma database"""
        query_parts = ["type: alert"]
        if alert_type:
            query_parts.append(f"alert_type: {alert_type}")
        if priority:
            query_parts.append(f"priority: {priority}")
            
        query = " AND ".join(query_parts)
        
        results = self.vector_store.similarity_search(
            query,
            k=100
        )
        
        alerts = []
        for doc in results:
            if hasattr(doc, 'metadata') and 'alert_data' in doc.metadata:
                try:
                    alert = json.loads(doc.metadata['alert_data'])
                    if date_from and alert.get('timestamp'):
                        if datetime.fromisoformat(alert['timestamp']) >= datetime.fromisoformat(date_from):
                            alerts.append(alert)
                    else:
                        alerts.append(alert)
                except json.JSONDecodeError:
                    continue
                    
        return alerts

    def process_visit_automation(self, visit_data: dict) -> Dict:
        """Process a visit and store automation data in Chroma"""
        automation_results = {
            "tasks": [],
            "alerts": [],
            "follow_ups": [],
            "notifications_sent": []
        }

        # Generate tasks
        tasks = self.analyze_meeting_outcome(visit_data)
        task_docs = []
        task_metadatas = []
        
        for task in tasks:
            task_dict = vars(task)
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(task_docs)}"
            task_dict['id'] = task_id
            
            task_doc = f"""
            Task ID: {task_id}
            Type: {task.task_type}
            Assigned To: {task.assigned_to}
            Priority: {task.priority}
            Status: {task.status}
            Description: {task.description}
            """
            
            task_docs.append(task_doc)
            task_metadatas.append({
                'task_id': task_id,
                'task_data': json.dumps(task_dict)
            })
            
            automation_results["tasks"].append(task_dict)

        # Store tasks in vector store
        if task_docs:
            self.vector_store.add_texts(
                texts=task_docs,
                metadatas=task_metadatas
            )

        # Process alerts
        if visit_data.get('marketIntel'):
            alerts = self.generate_competitor_alerts(visit_data['marketIntel'])
            alert_docs = []
            alert_metadatas = []
            
            for alert in alerts:
                alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(alert_docs)}"
                alert['id'] = alert_id
                
                alert_doc = f"""
                Alert ID: {alert_id}
                Type: {alert['type']}
                Priority: {alert['priority']}
                Description: {alert['description']}
                """
                
                alert_docs.append(alert_doc)
                alert_metadatas.append({
                    'alert_id': alert_id,
                    'alert_data': json.dumps(alert)
                })
                
                automation_results["alerts"].append(alert)

            # Store alerts in vector store
            if alert_docs:
                self.vector_store.add_texts(
                    texts=alert_docs,
                    metadatas=alert_metadatas
                )

        # Process follow-ups
        follow_ups = self.schedule_follow_ups(visit_data)
        follow_up_docs = []
        follow_up_metadatas = []
        
        for follow_up in follow_ups:
            follow_up_id = f"followup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(follow_up_docs)}"
            follow_up['id'] = follow_up_id
            
            follow_up_doc = f"""
            Follow-up ID: {follow_up_id}
            Client: {follow_up['client']}
            Date: {follow_up['date']}
            Action Items: {', '.join(follow_up['action_items'])}
            """
            
            follow_up_docs.append(follow_up_doc)
            follow_up_metadatas.append({
                'follow_up_id': follow_up_id,
                'follow_up_data': json.dumps(follow_up)
            })
            
            automation_results["follow_ups"].append(follow_up)

        # Store follow-ups in vector store
        if follow_up_docs:
            self.vector_store.add_texts(
                texts=follow_up_docs,
                metadatas=follow_up_metadatas
            )

        return automation_results