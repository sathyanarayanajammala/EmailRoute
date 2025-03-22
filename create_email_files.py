import json
import os
from pathlib import Path

def create_email_files():
    # Read the JSON file
    with open('email-config/email.json', 'r') as file:
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = Path('email_templates')
    output_dir.mkdir(exist_ok=True)
    
    def create_email_file(template_data, parent_path=''):
        for key, value in template_data.items():
            # Create subdirectory for each category
            current_path = output_dir / parent_path / key
            current_path.mkdir(exist_ok=True, parents=True)
            
            if isinstance(value, dict):
                if 'body' in value:
                    # This is an email template
                    filename = current_path / 'email.txt'
                    
                    # Create email content with metadata
                    email_content = f"""From: {value.get('from', 'N/A')}
To: {value.get('to', 'N/A')}
CC: {value.get('cc', 'N/A')}
Subject: {value.get('subject', 'N/A')}
{'='*50}

{value['body']}

{'='*50}
Attachments:
"""
                    # Add attachment information if present
                    if 'attachments' in value:
                        for attachment in value['attachments']:
                            email_content += f"\n- {attachment['filename']}"
                            if 'description' in attachment:
                                email_content += f"\n  Description: {attachment['description']}"
                    
                    # Write to file
                    with open(filename, 'w', encoding='utf-8') as email_file:
                        email_file.write(email_content)
                    
                    print(f"Created: {filename}")
                else:
                    # Recurse into nested directories
                    create_email_file(value, current_path)
    
    # Start creation from emailTemplates
    create_email_file(data['emailTemplates'])
    print("\nEmail files creation completed!")

if __name__ == "__main__":
    create_email_files()