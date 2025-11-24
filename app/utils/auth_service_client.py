import logging
import httpx
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthServiceClientError(Exception):
    """Raised when the auth service client cannot complete its task."""


class AuthServiceClient:
    """
    Client for interacting with the auth service.
    
    This class provides methods to fetch user details from the auth service
    using email addresses.
    """
    
    async def fetch_users_by_emails(self, email_list: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch user details from auth service by email addresses.
        
        Args:
            email_list: List of email addresses to fetch user details for
            
        Returns:
            List of user objects from auth service
            
        Raises:
            AuthServiceClientError: If auth service call fails or returns invalid data
        """
        if not email_list:
            return []
        
        if not settings.AUTH_SERVICE_URL:
            raise AuthServiceClientError("AUTH_SERVICE_URL not configured")
        
        try:
            url = f"{settings.AUTH_SERVICE_URL}/auth-service/api/v1/users/bulk-fetch"
            logger.info(f"Fetching user details for {len(email_list)} emails from auth service at {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json={"emailIds": email_list},
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") != "success":
                    raise AuthServiceClientError(f"Auth service returned non-success status: {data.get('status')}")
                
                users = data.get("data", {}).get("users", [])
                logger.info(f"Successfully fetched {len(users)} users from auth service")
                
                return users
                
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error when calling auth service: {exc.response.status_code} - {exc.response.text}")
            raise AuthServiceClientError(f"Auth service HTTP error: {exc.response.status_code}") from exc
            
        except httpx.RequestError as exc:
            logger.error(f"Request error when calling auth service: {exc}")
            raise AuthServiceClientError(f"Auth service request error: {exc}") from exc
            
        except Exception as exc:
            logger.error(f"Unexpected error when calling auth service: {exc}")
            raise AuthServiceClientError(f"Unexpected auth service error: {exc}") from exc


    def create_user_email_mapping(self, users: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Create a mapping of email to user data for easy lookup.
        
        Args:
            users: List of user objects from auth service
            
        Returns:
            Dictionary mapping email addresses to user objects
        """
        return [{ "email": user.get("email"), "id": user.get("id"), "name": user.get("name") } for user in users if user.get("email")]