import os
import json
import base64
from typing import List, Dict, Any
from pathlib import Path
import io
import pandas as pd
# Import CrewAI components
from crewai import Agent, Task, Crew, Process,LLM
from crewai.tasks import TaskOutput
# Document processing libraries
import PyPDF2
import docx
import pytesseract
from PIL import Image
from email.parser import Parser
from email.policy import default
from dotenv import load_dotenv
# Configure Gemini Pro as the LLM for CrewAI
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pathlib import Path
from time import sleep
# Set up environment
#os.environ["GEMINI_API_KEY"] = "AIzaSyDIFkAdvg2CONJxa4W8wHo7z98-PcR5IJY"  # Replace with actual API key



# Load and configure API
load_dotenv()

gemini_pro = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key="AIzaSyCJ3y0JCD0NaZ8zmvqgC7SrC3vFEsjdwgU",
    # Additional recommended parameters
    max_tokens=1024,        # Control response length
    top_p=0.95,            # Nucleus sampling parameter
    top_k=40,              # Limit vocabulary diversity
    presence_penalty=0.2,   # Reduce repetition
    frequency_penalty=0.3,  # Encourage diverse language
    context_window=8192,    # Maximum context length
    streaming=True         # Enable streaming responses
)

# Initialize Gemini Pro


# Document Processing Utilities
class DocumentProcessor:
    """Utility class for processing emails and attachments"""
    
    @staticmethod
    def extract_text_from_email(email_content: str) -> Dict[str, Any]:
        """Extract text and attachments from email content."""
        email_parser = Parser(policy=default)
        parsed_email = email_parser.parsestr(email_content)
        
        result = {
            "subject": parsed_email.get("subject", ""),
            "from": parsed_email.get("from", ""),
            "to": parsed_email.get("to", ""),
            "date": parsed_email.get("date", ""),
            "body": "",
            "attachments": []
        }
        
        # Extract body
        if parsed_email.is_multipart():
            for part in parsed_email.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    result["body"] = part.get_payload(decode=True).decode()
                    break
        else:
            result["body"] = parsed_email.get_payload(decode=True).decode()
        
        # Extract attachments
        if parsed_email.is_multipart():
            for part in parsed_email.walk():
                if part.get_content_maintype() != 'multipart' and part.get('Content-Disposition') is not None:
                    filename = part.get_filename()
                    if filename:
                        file_content = part.get_payload(decode=True)
                        result["attachments"].append({
                            "filename": filename,
                            "content": file_content,
                            "content_type": part.get_content_type()
                        })
        
        return result
    
    @staticmethod
    def extract_text_from_attachment(attachment: Dict[str, Any]) -> str:
        """Extract text from attachment based on file type."""
        filename = attachment["filename"]
        content = attachment["content"]
        
        # Determine file type
        file_extension = Path(filename).suffix.lower()
        
        try:
            # PDF
            if file_extension == ".pdf":
                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                return text
            
            # Word Document
            elif file_extension in [".doc", ".docx"]:
                docx_file = io.BytesIO(content)
                doc = docx.Document(docx_file)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Image (requires OCR)
            elif file_extension in [".png", ".jpg", ".jpeg"]:
                image = Image.open(io.BytesIO(content))
                return pytesseract.image_to_string(image)
            
            # Text files
            elif file_extension in [".txt", ".csv"]:
                return content.decode("utf-8")
            
            else:
                return f"[Unsupported file type: {file_extension}]"
                
        except Exception as e:
            return f"[Error processing attachment: {str(e)}]"

# Data loading utilities
def load_team_skills(file_path="team_skills.csv") -> pd.DataFrame:
    """Load team skills from CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading team skills: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "employee_id", "name", "department", "role", 
            "request_types", "sub_request_types" 
            "technical_skills", "domain_knowledge", "languages",
            "workload", "availability"
        ])

# Create specialized agents
content_analyzer_agent = Agent(
    role="Email Content Analyzer",
    goal="Accurately analyze and summarize email content and attachments",
    backstory="""You are an expert at reading and interpreting emails and documents.
    You have years of experience in natural language processing and can quickly identify
    the key points in any communication.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini_pro
)

intent_classifier_agent = Agent(
    role="Intent Classifier",
    goal="Correctly identify the intent and classify emails by type and priority",
    backstory="""You are specialized in understanding user intent in communications.
    You can determine what people are asking for and categorize requests appropriately.
    You have extensive experience working in customer service triage.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini_pro
)

attribute_extractor_agent = Agent(
    role="Attribute Extractor",
    goal="Extract key information and attributes from emails with high accuracy",
    backstory="""You are a detail-oriented analyst with a talent for spotting and
    extracting specific information from text. You know exactly what data points
    are important in business communications.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini_pro
)

assignment_specialist_agent = Agent(
    role="Request Assignment Specialist",
    goal="Match requests to the most appropriate team members based on skills and availability",
    backstory="""You are a resource allocation expert with deep knowledge of team structures.
    You understand people's skills and workloads, and can make optimal assignments
    for handling different types of requests.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini_pro
)

# Define tasks for the email processing pipeline
def create_summarization_task(email_data):
    """Create a task for summarizing email content and attachments"""
    
    # Format attachment information
    attachment_info = ""
    for idx, attachment in enumerate(email_data.get("attachments", [])):
        if "extracted_text" in attachment and attachment["extracted_text"]:
            attachment_info += f"\n\nATTACHMENT {idx+1}: {attachment['filename']}\n"
            attachment_info += attachment["extracted_text"][:3000]  # Limit to first 3000 chars
    
    return Task(
        description=f"""
        Analyze and summarize the following email content:
        
        SUBJECT: {email_data["subject"]}
        FROM: {email_data["from"]}
        TO: {email_data["to"]}
        DATE: {email_data["date"]}
        
        EMAIL BODY:
        {email_data["body"]}
        
        {attachment_info if attachment_info else ""}
        
        Provide a comprehensive summary that captures the key points, requests, and important details.
        Format your response as a JSON with a 'summary' field containing your summary text.
        """,
        agent=content_analyzer_agent,
        expected_output="A JSON object with a comprehensive summary of the email content"
    )

def create_intent_classification_task(email_data, summary):
    """Create a task for identifying intent and classifying the email"""
    return Task(
        description=f"""
        Analyze the following email and classify its primary intent:
        
        SUBJECT: {email_data["subject"]}
        FROM: {email_data["from"]}
        
        EMAIL SUMMARY:
        {summary}
        
        Based on the email content, determine:
        1. Primary intent (choose one): Customer Support, Technical Issue, Account Management, Billing Question, Feature Request, Bug Report, General Inquiry, Sales Lead, Partnership Opportunity, Complaint, Urgent Issue, Other
        2. Priority level (choose one): Low, Medium, High, Critical
        3. Response time expectation (choose one): 24 hours, 48 hours, 72 hours, 1 week
        4. Department(s) responsible (can be multiple): Loan Servicing,Loan Closing,Account Management,Treasury Management,Financial Operations,International Banking,IT,Support,Legal,Finance,Product,HR,Marketing,Engineering
        5. Request type or Sub_request_types responsible (can be multiple): Adjustment, AU Transfer,IT,Support,Closing Notice,commitment change,Fee Payment,Money Movement-Inbound,Money Movement - Outbound,Foreign Currency,Principal, Interest,
          Principal + Interest, Principal+Interest+Fee, Timebound,Ongoing Fee, Letter of Credit Fee,	Cashless Roll, Decrease, Increase,Reallocation Fees, 
          Amendment Fees, Reallocation Principal
        6. Sentiment: Is the sender satisfied, neutral, or dissatisfied?
        
        Respond with a valid JSON object containing these fields.
        """,
        agent=intent_classifier_agent,
        expected_output="A JSON object with intent classification details"
    )

def create_attribute_extraction_task(email_data, summary):
    """Create a task for extracting key attributes from the email"""
    return Task(
        description=f"""
        Extract key attributes from the following email:
        
        SUBJECT: {email_data["subject"]}
        FROM: {email_data["from"]}
        
        EMAIL SUMMARY:
        {summary}
        
        Extract the following attributes (leave empty if not found):
        1. customer_name: Customer/Client name
        2. account_id: Account ID/number (if mentioned)
        3. product_service: Product or service mentioned
        4. issue_description: Issue description
        5. deadline: Any deadlines mentioned (in YYYY-MM-DD format)
        6. amount: Any amounts/prices mentioned
        7. requested_action: Specific action requested
        8. previous_communication: References to previous communications
        9. contact_info: Contact information provided (phone, email)
        
        Respond with a valid JSON object containing these fields.
        """,
        agent=attribute_extractor_agent,
        expected_output="A JSON object with extracted attributes"
    )

def create_assignment_task(intent_data, attributes, team_data):
    """Create a task for assigning the request to a team member"""
    team_json = json.dumps(team_data)
    
    return Task(
        description=f"""
        Given the following request details and team information, determine the most appropriate person to handle this request:
        
        REQUEST DETAILS:
        {json.dumps(intent_data, indent=2)}
        
        EXTRACTED ATTRIBUTES:
        {json.dumps(attributes, indent=2)}
        
        TEAM INFORMATION:
        {team_json}
        
        Select the best match based on:
        1. alignment with request type or sub_request_types
        2. Required skills and domain knowledge
        3. Current workload and availability
        4. Priority level of the request
        
        Provide your recommendation as a JSON object with these fields:
        1. assigned_employee_id: ID of the assigned employee
        2. assigned_employee_name: Name of the assigned employee
        3. rationale: Briefly explain why this person is the best match
        4. confidence_score: A score between 0-100 indicating confidence in this assignment
        5. alternative_assignee: A backup person if the primary is unavailable
        """,
        agent=assignment_specialist_agent,
        expected_output="A JSON object with assignment recommendation"
    )

# Main Email Triage System using CrewAI
class CrewAIEmailTriageSystem:
    """Email triage system implemented using CrewAI framework"""
    
    def __init__(self, team_skills_file="team_skills.csv"):
        """Initialize the email triage system"""
        self.doc_processor = DocumentProcessor()
        self.team_skills = load_team_skills(team_skills_file)
        self.team_data = self.team_skills.to_dict(orient="records")
    
    def process_email(self, email_content: str) -> Dict[str, Any]:
        """Process a single email and return the analysis results"""
        # Extract email content
        email_data = self.doc_processor.extract_text_from_email(email_content)
        
        # Process attachments
        for attachment in email_data.get("attachments", []):
            attachment["extracted_text"] = self.doc_processor.extract_text_from_attachment(attachment)
        
        # Create tasks
        summarization_task = create_summarization_task(email_data)
        
        # Create the crew with sequential processing
        initial_crew = Crew(
            agents=[content_analyzer_agent],
            tasks=[summarization_task],
            process=Process.sequential
        )
        
        # Run the initial crew to get the summary
        summary_result = initial_crew.kickoff()
        
        try:
            summary_data = json.loads(summary_result.raw.
                                      replace('```json', '').replace('```', ''))
            summary = summary_data.get("summary", "")
        except json.JSONDecodeError:
            # Fallback if output is not valid JSON
            summary = summary_result
        
        # Create tasks for the next steps
        intent_task = create_intent_classification_task(email_data, summary)
        attribute_task = create_attribute_extraction_task(email_data, summary)
        
        # Create a crew for parallel processing of intent and attribute extraction
        analysis_crew = Crew(
            agents=[intent_classifier_agent, attribute_extractor_agent],
            tasks=[intent_task, attribute_task],
            process=Process.sequential
        )
        
        # Run the analysis crew
        analysis_results = analysis_crew.kickoff()
        
        # Parse results
        try:
            intent_data = json.loads(analysis_results.tasks_output[0].raw
                                     .replace('```json', '').replace('```', ''))
        except (json.JSONDecodeError, IndexError):
            intent_data = {"error": "Failed to parse intent classification"}
        
        try:
            attributes = json.loads(analysis_results.tasks_output[1].raw.
                                    replace('```json', '').replace('```', ''))
        except (json.JSONDecodeError, IndexError):
            attributes = {"error": "Failed to parse attribute extraction"}
        
        # Create assignment task
        assignment_task = create_assignment_task(intent_data, attributes, self.team_data)
        
        # Create assignment crew
        assignment_crew = Crew(
            agents=[assignment_specialist_agent],
            tasks=[assignment_task],
            process=Process.sequential
        )
        
        # Run the assignment crew
        assignment_result = assignment_crew.kickoff()
        
        try:
            assignment = json.loads(assignment_result.raw.
                                    replace('```json', '').replace('```', ''))
        except json.JSONDecodeError:
            assignment = {"error": "Failed to parse assignment recommendation"}
        
        # Compile results
        return {
            "email_metadata": {
                "subject": email_data["subject"],
                "from": email_data["from"],
                "to": email_data["to"],
                "date": email_data["date"],
            },
            "summary": summary,
            "intent_classification": intent_data,
            "extracted_attributes": attributes,
            "assignment": assignment,
            "attachments": [
                {"filename": att["filename"], "content_type": att["content_type"]}
                for att in email_data.get("attachments", [])
            ]
        }
def extract_email_bodies():
    # Read the JSON file
    with open('email-config/email.json', 'r') as file:
        data = json.load(file)
    
    # Dictionary to store all email bodies
    email_bodies = {}
    
    # Recursively extract bodies from the email templates
    def extract_bodies(templates, parent_key=''):
        for key, value in templates.items():
            current_key = f"{parent_key}/{key}" if parent_key else key
            
            if isinstance(value, dict):
                if 'body' in value:
                    # If there's a body field directly in this dict
                    email_bodies[current_key] = value['body']
                else:
                    # Recurse into nested dictionaries
                    extract_bodies(value, current_key)
    
    # Start extraction from emailTemplates
    extract_bodies(data['emailTemplates'])
    
    return email_bodies
# Example usage
def process_email_samples():
    # Get all email files from the email_Samples directory
    samples_dir = Path('email_Samples')
    email_files = [f for f in samples_dir.glob('*.txt')]
    
     # Find and filter out duplicates
    unique_files, duplicate_files = find_duplicate_emails(email_files)
    
     # Get current working directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
     # Print duplicate information
    if duplicate_files:
        print("\nFound duplicate emails:")
        for duplicate, original in duplicate_files:
            print(f"  - Duplicate: {duplicate.relative_to(current_dir)}")
            print(f"    Original: {original.relative_to(current_dir)}")
        print(f"\nProcessing {len(unique_files)} unique emails...")
    else:
        print("\nNo duplicate emails found.")

    if not unique_files:
        print("No unique email files to process!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path('output_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create a single output file for all results
    output_file = output_dir / 'email_analysis_results.txt'
    # Create the triage system
    system = CrewAIEmailTriageSystem()
    with open(output_file, 'w', encoding='utf-8') as out_file:
    # Process each email file
        for email_file in unique_files:
            result_text =""
                
            try:
                # Read the email content
                with open(email_file, 'r', encoding='utf-8') as file:
                    email_content = email_file.read_text()
                result_text = f"\nProcessing: {email_file.name}\n"
                result_text += "="*50 + "\n"
                # Process the email through the triage system
                results = system.process_email(email_content)
                
                # Format results
                result_text += "======= EMAIL TRIAGE RESULTS =======\n"
                result_text += f"SUBJECT: {results['email_metadata']['subject']}\n"
                result_text += f"FROM: {results['email_metadata']['from']}\n"
                result_text += "\nSUMMARY:\n"
                result_text += f"{results['summary']}\n"
                result_text += "\nINTENT CLASSIFICATION:\n"
                result_text += f"{json.dumps(results['intent_classification'], indent=2)}\n"
                result_text += "\nEXTRACTED ATTRIBUTES:\n"
                result_text += f"{json.dumps(results['extracted_attributes'], indent=2)}\n"
                result_text += "\nASSIGNMENT:\n"
                result_text += f"{json.dumps(results['assignment'], indent=2)}\n"
                result_text += "\n" + "="*50 + "\n"
                # Write to file and also print to console
                out_file.write(result_text)
                print(result_text)  # Optional: keep console output as well
                sleep(30)
            except Exception as e:
                result_text += f"Error processing {email_file.name}: {str(e)}\n"
                print(result_text)
                result_text += "="*50 + "\n"
def find_duplicate_emails(email_files: list) -> tuple[list, list]:
    """
    Find duplicate emails by comparing their content.
    Returns a tuple of (unique_files, duplicate_files)
    """
    content_hash_map = {}  # Maps content hash to file path
    unique_files = []
    duplicate_files = []

    for email_file in email_files:
        try:
            # Read content and create a hash
            content = email_file.read_text(encoding='utf-8')
            content_hash = hash(content)

            if content_hash in content_hash_map:
                # This is a duplicate
                original_file = content_hash_map[content_hash]
                duplicate_files.append((email_file, original_file))
            else:
                # This is a unique file
                content_hash_map[content_hash] = email_file
                unique_files.append(email_file)

        except Exception as e:
            print(f"Error reading file {email_file}: {str(e)}")
            continue

    return unique_files, duplicate_files
      
        
def main():
    # Create sample team skills CSV if it doesn't exist
    if not os.path.exists("team_skills.csv"):
        with open('email-config/types.json', 'r') as file:
            team_data = json.load(file)['employees']
        pd.DataFrame(team_data).to_csv("team_skills.csv", index=False)
    
    # Process all email samples
    process_email_samples()
if __name__ == "__main__":
    main()
