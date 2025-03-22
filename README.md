# Email Triage System

An intelligent email processing system for loan servicing operations using CrewAI and LangChain with emini-2.0-flash LLM.

## Output Details

### Output File Location
The system generates analysis results in: `output_results/email_analysis_results.txt`

### Output Format Structure
Each email analysis contains:

```json
{
  "email_metadata": {
    "subject": "Loan Adjustment Notification",
    "from": "adjustments@loanservicing.com",
    "to": "john.smith@example.com",
    "date": "2025-03-15"
  },
  "summary": "Detailed summary of email content...",
  "intent_classification": {
    "intent": "Account Management",
    "priority": "Medium",
    "response_time": "48 hours",
    "departments": ["Finance", "Support"],
    "sentiment": "Neutral"
  },
  "extracted_attributes": {
    "customer_name": "John Smith",
    "account_id": "L-45872-93A",
    "product_service": "Loan",
    "issue_description": "Interest rate recalculation",
    "deadline": null,
    "amount": "$258.75",
    "requested_action": "Review adjustment",
    "contact_info": "(800) 555-1234"
  },
  "assignment": {
    "assigned_employee_id": "E001",
    "assigned_employee_name": "Alice Johnson",
    "rationale": "Assignment reasoning...",
    "confidence_score": 90,
    "alternative_assignee": {
      "employee_id": "E003",
      "employee_name": "Carol White"
    }
  }
}
```

### Sample Email Types

1. **Loan Adjustments**
```
From: adjustments@loanservicing.com
Subject: Loan Adjustment Notification
Content: Interest rate recalculation notices, principal adjustments
```

2. **Money Movement**
```
From: internationaltransfers@loanservicing.com
Subject: Outbound Foreign Currency Transfer
Content: Currency transfers, exchange rates, transfer confirmations
```

3. **Commitment Changes**
```
From: commitments@loanservicing.com
Subject: Commitment Increase/Decrease Notification
Content: Changes in commitment amounts, terms modifications
```

4. **Fee Processing**
```
From: closings@loanservicing.com
Subject: Amendment Fees/Fee Reallocation
Content: Fee calculations, payment schedules, reallocations
```

### Processing Examples

1. **Interest Payment Processing**
- Input: Email about inbound interest payment
- Output: Assignment to Treasury Management specialist
- Confidence Score: 85-95%
- Response Time: Based on payment priority

2. **Loan Adjustment Processing**
- Input: Interest rate recalculation notice
- Output: Assignment to Loan Adjustment Specialist
- Confidence Score: 90-95%
- Response Time: 24-48 hours

3. **Foreign Currency Transfers**
- Input: International transfer notification
- Output: Assignment to Foreign Exchange specialist
- Confidence Score: 85-95%
- Response Time: Based on transfer urgency

## Model Configuration

The system uses Gemini 2.0 Flash model for enhanced processing:
```python
model="gemini/gemini-2.0-flash"
temperature=0.7
```

## Rate Limiting
- 120-second delay between email processing
- Prevents system overload
- Ensures accurate analysis

## Error Handling
The system generates error logs for:
- File processing failures
- Invalid email formats
- Attachment processing issues
- Assignment conflicts

## Detailed Features

### 1. Email Content Processing
- **Document Processing**
  - Handles multiple email formats and attachments
  - Supports file types: PDF, Word (.doc/.docx), Images (PNG/JPG), Text files
  - OCR capability for image-based documents using pytesseract
  - Maintains email metadata (subject, from, to, date)

### 2. Intelligent Analysis System
The system employs four specialized AI agents:

#### a. Content Analyzer Agent
- Provides comprehensive email summaries
- Analyzes both main content and attachments
- Identifies key discussion points
- Extracts critical information from attachments

#### b. Intent Classifier Agent
Categorizes emails by:
- Primary Intent:
  - Customer Support
  - Technical Issue
  - Account Management
  - Billing Question
  - Feature Request
  - Bug Report
  - General Inquiry
  - Sales Lead
  - Partnership Opportunity
  - Complaint
  - Urgent Issue
- Priority Level (Low/Medium/High/Critical)
- Response Time Requirements (24h/48h/72h/1 week)
- Department Assignment (IT/Sales/Support/Legal/Finance/Product/HR/Marketing/Engineering)
- Sender Sentiment Analysis

#### c. Attribute Extractor Agent
Automatically extracts:
- Customer/Client names
- Account IDs
- Product/Service references
- Issue descriptions
- Deadlines
- Financial amounts
- Requested actions
- Previous communication references
- Contact information

#### d. Assignment Specialist Agent
Performs intelligent task routing based on:
- Department alignment
- Required skills and domain knowledge
- Current workload and availability
- Request priority
- Provides assignment recommendations with:
  - Primary assignee selection
  - Assignment confidence score
  - Alternative assignee suggestion
  - Assignment rationale

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
Produces structured analysis results including:
- Email metadata
- Comprehensive content summary
- Intent classification details
- Extracted attributes
- Assignment recommendations
- Attachment processing results

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
