import os
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from dataclasses import dataclass
from typing import Optional
from sales_rag import SalesRAG
from sales_automation import SalesAutomation
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import json
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure main application logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        'logs/app.log', 
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    
    return app_logger

# Initialize logger
logger = setup_logging()

# Verify required environment variables
required_env_vars = ['GROQ_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Data Models
@dataclass
class SalesVisit:
    clientName: str
    visitDate: str
    location: str
    productsDiscussed: str
    meetingOutcome: str
    orderDetails: Optional[str]
    nextSteps: str
    marketIntel: Optional[str]
    remarks: Optional[str]

@dataclass
class PerformanceQuery:
    startDate: str
    endDate: str
    salesPersonId: str

# Initialize systems
try:
    rag_system = SalesRAG()
    automation_system = SalesAutomation()
    logger.info("Successfully initialized RAG and Automation systems")
except Exception as e:
    logger.error(f"Error initializing systems: {str(e)}")
    raise

# Route for serving the index page
@app.route("/")
def index():
    logger.info("Serving index page")
    return render_template('index.html')

# API Endpoints
@app.route("/api/sales-report", methods=["POST"])
def submit_sales_report():
    try:
        visit_data = request.get_json()
        if not visit_data:
            logger.warning("Empty sales report submission")
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400

        logger.info(f"Received sales report submission for client: {visit_data.get('clientName', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['clientName', 'visitDate', 'location', 'productsDiscussed', 'meetingOutcome', 'nextSteps']
        missing_fields = [field for field in required_fields if field not in visit_data]
        if missing_fields:
            logger.warning(f"Missing required fields in sales report: {missing_fields}")
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Process with RAG system
        logger.debug("Processing with RAG system")
        rag_result = rag_system.process_visit_report(visit_data)
        
        # Process with automation system
        logger.debug("Processing with automation system")
        automation_result = automation_system.process_visit_automation(visit_data)
        
        # Combine results
        result = {
            "status": "success",
            "rag_analysis": rag_result,
            "automation": automation_result
        }
        
        logger.info("Successfully processed sales report")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing sales report: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/schedule/<salesperson_id>", methods=["GET"])
def get_schedule(salesperson_id):
    try:
        date = request.args.get('date')
        logger.info(f"Fetching schedule for salesperson {salesperson_id} on {date}")
        
        if not date:
            logger.warning("Schedule request missing date parameter")
            return jsonify({
                "status": "error",
                "message": "Date parameter is required"
            }), 400
        
        try:
            # Validate date format
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid date format provided: {date}")
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400
        
        schedule = rag_system.get_daily_schedule(salesperson_id, date)
        logger.info(f"Successfully retrieved schedule for salesperson {salesperson_id}")
        return jsonify(schedule)
    except Exception as e:
        logger.error(f"Error retrieving schedule: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Chat request missing message")
            return jsonify({
                "status": "error",
                "message": "Message is required"
            }), 400

        # Get conversation history if provided
        conversation_history = data.get('history', [])
        
        # Initialize chat model with environment variables
        chat_model = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=float(os.getenv('MODEL_TEMPERATURE', '0.3')),
            max_tokens=int(os.getenv('MAX_TOKENS', '4096')),
            max_retries=int(os.getenv('RETRY_ATTEMPTS', '2')),
            timeout=int(os.getenv('TIMEOUT_SECONDS', '30'))
        )

        # Initialize conversation memory
        memory = ConversationBufferMemory()
        
        # Add conversation history to memory
        for message in conversation_history:
            if message['role'] == 'user':
                memory.save_context(
                    {"input": message['content']}, 
                    {"output": ""}
                )
            elif message['role'] == 'assistant':
                memory.save_context(
                    {"input": ""}, 
                    {"output": message['content']}
                )

        # Use the existing vector store from rag_system
        relevant_docs = rag_system.vector_store.similarity_search(
            data['message'],
            k=int(os.getenv('RELEVANT_DOCS_COUNT', '5'))
        )

        # Extract and format context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""Use the following context from our knowledge base to help answer the question. If the context isn't relevant, you can ignore it and answer based on your general knowledge.

Context:
{context}

Question: {data['message']}

Please provide a helpful and accurate response."""

        # Create conversation chain with enhanced prompt
        conversation = ConversationChain(
            llm=chat_model,
            memory=memory,
            verbose=True
        )

        # Get response with context-enhanced prompt
        response = conversation.predict(input=enhanced_prompt)
        
        logger.info(f"Generated context-enhanced chat response for message: {data['message'][:50]}...")
        
        return jsonify({
            "status": "success",
            "response": response,
            "conversation_id": data.get('conversation_id', str(uuid.uuid4()))
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Sorry, there was an error processing your request. Please try again."
        }), 500

@app.route("/api/performance", methods=["POST"])
def get_performance():
    try:
        query = request.get_json()
        if not query:
            logger.warning("Empty performance query")
            return jsonify({
                "status": "error",
                "message": "No query data provided"
            }), 400

        logger.info(f"Processing performance query for salesperson {query.get('salesPersonId', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['startDate', 'endDate', 'salesPersonId']
        missing_fields = [field for field in required_fields if field not in query]
        if missing_fields:
            logger.warning(f"Missing required fields in performance query: {missing_fields}")
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Validate date formats
        try:
            datetime.strptime(query['startDate'], '%Y-%m-%d')
            datetime.strptime(query['endDate'], '%Y-%m-%d')
        except ValueError:
            logger.warning("Invalid date format in performance query")
            return jsonify({
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD"
            }), 400
        
        insights = rag_system.get_performance_insights(query)
        logger.info("Successfully retrieved performance insights")
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error retrieving performance insights: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        "status": "error", 
        "message": "Resource not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}", exc_info=True)
    return jsonify({
        "status": "error", 
        "message": "Internal server error"
    }), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask application on port {port} with debug={debug}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )