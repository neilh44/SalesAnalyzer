import os
from datetime import datetime
from typing import Dict
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import json

class SalesRAG:
    def __init__(self):
        # Initialize Groq LLM with recommended settings
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=4096,
            max_retries=2,
            timeout=30
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        os.makedirs("./data/chroma_db", exist_ok=True)
        
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )

    def process_visit_report(self, visit_data: dict) -> Dict:
        """Process and analyze a sales visit report"""
        visit_text = f"""
        Sales Visit Report
        Client: {visit_data['clientName']}
        Date: {visit_data['visitDate']}
        Location: {visit_data['location']}
        
        Products Discussed: {visit_data['productsDiscussed']}
        Meeting Outcome: {visit_data['meetingOutcome']}
        
        Order Details: {visit_data.get('orderDetails', 'No orders placed')}
        Next Steps: {visit_data['nextSteps']}
        
        Market Intelligence: {visit_data.get('marketIntel', 'No market intel provided')}
        Additional Remarks: {visit_data.get('remarks', 'No additional remarks')}
        """
        
        texts = self.text_splitter.create_documents([visit_text])
        self.vector_store.add_documents(texts)
        
        analysis_query = f"""
        Analyze this sales visit report and provide:
        1. Key achievements or progress made
        2. Critical follow-up items
        3. Potential opportunities identified
        4. Recommended next actions
        5. Any red flags or concerns
        """
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        analysis = qa_chain.run(analysis_query)
        
        return {
            "status": "success",
            "report_id": f"VISIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "analysis": analysis
        }

    def get_performance_insights(self, query: dict) -> Dict:
        """Generate performance insights for a period"""
        insight_query = f"""
        Analyze sales performance for {query['salesPersonId']} 
        from {query['startDate']} to {query['endDate']}.
        Focus on:
        1. Visit completion rate
        2. Order conversion rate
        3. Key achievements
        4. Areas for improvement
        5. Client engagement levels
        """
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        insights = qa_chain.run(insight_query)
        
        return {
            "status": "success",
            "salesperson_id": query['salesPersonId'],
            "period": f"{query['startDate']} to {query['endDate']}",
            "insights": insights
        }

    def get_daily_schedule(self, salesperson_id: str, date: str) -> Dict:
        """Generate daily schedule and insights"""
        schedule_query = f"""
        Based on historical data and pending follow-ups,
        provide a recommended schedule for {salesperson_id} on {date}.
        Include:
        1. Priority clients to visit
        2. Key discussion points
        3. Preparation needed
        4. Potential opportunities
        """
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        schedule = qa_chain.run(schedule_query)
        
        return {
            "status": "success",
            "date": date,
            "salesperson_id": salesperson_id,
            "schedule": json.loads(schedule)
        }