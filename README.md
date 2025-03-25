# Email Triage System

An intelligent email processing system for loan servicing operations using CrewAI and LangChain with emini-2.0-flash LLM.

## Model Configuration

### LLM Configuration
The system uses Gemini 2.0 Flash model with optimized hyperparameters for email processing:

```python
gemini_pro = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.4,        # Optimized for consistent classification
    max_tokens=1024,        # Balanced for email analysis
    top_p=0.9,             # Focused response generation
    top_k=40,              # Controlled vocabulary diversity
    presence_penalty=0.1,   # Minimal repetition control
    frequency_penalty=0.1,  # Professional language maintenance
    context_window=8192,    # Large context for attachments
    streaming=True         # Real-time processing
)
```

### Hyperparameter Optimization

#### Core Parameters
- **Temperature (0.4)**
  - Optimized for consistent email classification
  - Reduces random variations in intent analysis
  - Maintains reliable attribute extraction

- **Max Tokens (1024)**
  - Sufficient for comprehensive email analysis
  - Handles detailed summaries and classifications
  - Approximately 750-800 words per response

- **Top P (0.9)**
  - Nucleus sampling for focused responses
  - Ensures relevant content generation
  - Maintains professional communication style

#### Advanced Controls
- **Top K (40)**
  - Controls vocabulary diversity
  - Optimized for business communication
  - Maintains consistent terminology

- **Presence/Frequency Penalties (0.1)**
  - Minimal repetition in responses
  - Maintains professional language
  - Ensures natural response flow

#### Processing Parameters
- **Context Window (8192)**
  - Handles long emails with attachments
  - Processes comprehensive document content
  - Supports multi-page analysis

- **Streaming (Enabled)**
  - Real-time processing feedback
  - Improved user experience
  - Progressive response generation

## Detailed Features

### 1. Email Content Processing
- **Document Processing**
  - Handles multiple email formats and attachments
  - Supports file types: PDF, Word (.doc/.docx), Images (PNG/JPG), Text files
  - OCR capability for image-based documents using pytesseract
  - Maintains email metadata (subject, from, to, date)

### 2. Request Type Classification

#### Primary Request Types
1. **Adjustment**
   - Loan term modifications
   - Rate adjustments
   - Payment schedule changes

2. **AU Transfer**
   - Administrative unit transfers
   - Portfolio reassignments

3. **Closing Notice**
   - Sub-types:
     - Reallocation Fees
     - Amendment Fees
     - Reallocation Principal
   - Documentation requirements
   - Fee processing

4. **Commitment Change**
   - Sub-types:
     - Cashless Roll
     - Decrease
     - Increase
   - Term modifications
   - Limit adjustments

5. **Money Movement-Inbound**
   - Sub-types:
     - Principal
     - Interest
     - Principal + Interest
     - Principal + Interest + Fee
     - Timebound
   - Payment processing
   - Fund allocation

### 3. Email Types Handled
Specializes in loan servicing operations including:
- Loan Adjustments
- Amendment Fee Processing
- Cashless Roll Commitments
- Credit Fee Payments
- Commitment Changes (Increase/Decrease)
- Inbound Payments
  - Principal
  - Interest
  - Fees
- Principal Reallocation
- Fee Reallocation

### 4. Output Generation
Produces structured analysis results in the following format:

```json
{
    "email_metadata": {
        "subject": "string",
        "from": "string",
        "to": "string",
        "date": "datetime"
    },
    "summary": "string",
    "intent_classification": {
        "primary_type": "string",
        "sub_type": "string",
        "confidence_score": "float",
        "priority_level": "string",
        "department": "string"
    },
    "extracted_attributes": {
        "customer_name": "string",
        "account_id": "string",
        "product_reference": "string",
        "issue_description": "string",
        "deadlines": "datetime",
        "financial_amounts": "float",
        "requested_actions": "string",
        "previous_communication": "string",
        "contact_info": "object"
    },
    "assignment": {
        "primary_assignee": "string",
        "confidence_score": "float",
        "alternative_assignee": "string",
        "assignment_rationale": "string",
        "department": "string",
        "priority": "string",
        "sla_deadline": "datetime"
    },
    
}
```

### 5. Error Handling & Reliability
- Robust error handling for file processing
- Fallback mechanisms for parsing failures
- Duplicate email detection
- Rate limiting (120-second delay between processes)
- Validation of output formats

### 6. Configuration & Customization
- Team skills management via CSV
- Configurable email templates
- Adjustable processing parameters
- Custom department routing rules
- Flexible output formatting

### 7. Integration Capabilities
- Uses standard email protocols
- JSON-based data exchange
- CSV support for team management
- Modular architecture for easy integration
- Environment variable configuration

## Technical Implementation

### AI Framework
- Uses CrewAI for agent orchestration
- Implements LangChain with Gemini Pro LLM
- Supports parallel and sequential processing
- Maintains agent state and context

### Processing Pipeline
1. Email Content Extraction
2. Attachment Processing
3. Content Analysis
4. Intent Classification
5. Attribute Extraction
6. Assignment Determination
7. Results Compilation

### Output Storage
- Generates detailed analysis in `output_results/email_analysis_results.txt`
- video recording is present output_results zip file. extract the zip for video recording.
- Maintains structured format for easy parsing
- Includes timestamps and processing metadata
- Preserves full analysis chain

## Use Cases

1. **Loan Processing Operations**
   - Interest rate adjustments
   - Payment processing
   - Commitment changes
   - Fee management

2. **Customer Service**
   - Request routing
   - Priority assignment
   - Response time management
   - Team workload balancing

3. **Document Management**
   - Automated filing
   - Content extraction
   - Information categorization
   - Attachment processing
