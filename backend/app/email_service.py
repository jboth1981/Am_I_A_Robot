import httpx
import os
import json

async def send_password_reset_email(email: str, reset_url: str):
    """Send password reset email via Resend"""
    
    api_key = os.getenv("RESEND_API_KEY")
    from_email = os.getenv("MAIL_FROM", "noreply@amiarobot.ca")
    
    url = "https://api.resend.com/emails"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "from": from_email,
        "to": [email],
        "subject": "Password Reset - Am I A Robot",
        "html": f"""
        <h2>Password Reset Request</h2>
        <p>You requested a password reset for your Am I A Robot account.</p>
        <p>Click the link below to reset your password:</p>
        <a href="{reset_url}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a>
        <p>This link will expire in 1 hour.</p>
        <p>If you didn't request this reset, please ignore this email.</p>
        """,
        "text": f"""
        Password Reset Request
        
        You requested a password reset for your Am I A Robot account.
        
        Click the link below to reset your password:
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you didn't request this reset, please ignore this email.
        """
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            print(f"Email sent successfully to {email}, ID: {result.get('id')}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error sending email: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise