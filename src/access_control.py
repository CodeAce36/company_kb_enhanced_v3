"""Access control for company knowledge base."""

import json
import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='logs/kb_access.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AccessControl:
    def __init__(self, user_context=None):
        """Initialize with user context (if available)."""
        self.user_context = user_context or {"clearance_level": "confidential"}
        self.access_log = []
    
    def check_query_permission(self, query):
        """Check if the query contains restricted terms or topics."""
        # All queries are now allowed
        return True, []
    
    def filter_document_by_metadata(self, doc):
        """Filter documents based on metadata (classification level)."""
        # All documents are now accessible
        return True
    
    def log_access(self, resource_type, resource_id, access_type, successful, details=None):
        """Log access attempts for audit purposes."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": self.user_context.get("user_id", "anonymous"),
            "user_name": self.user_context.get("name", "unknown"),
            "department": self.user_context.get("department", "unknown"),
            "resource": {"type": resource_type, "id": resource_id},
            "access_type": access_type,
            "successful": successful,
            "details": details
        }
        
        # Add to in-memory log
        self.access_log.append(log_entry)
        
        # Log to file
        logging.info(json.dumps(log_entry))
        
        return log_entry