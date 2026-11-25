import requests
import cloudinary
import cloudinary.uploader
from config import CHANNEL_ACCESS_TOKEN, USER_ID, LINE_API_URL

# ==============================
# Function: Send LINE Message
# ==============================

def send_line_message(message, image_path=None):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
    }

    data = {
        "to": USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    
    if image_path:
        try:
            upload_result = cloudinary.uploader.upload(image_path)
            image_url = upload_result.get("secure_url")
            
            data["messages"].append({
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url
            })
        except Exception as e:
            print("เกิดข้อผิดพลาดในการอัปโหลดภาพไปยัง Cloudinary")
            print(e)

    data = {
        "to": USER_ID,
        "messages": data["messages"]
    }

    try:

        response = requests.post(
            LINE_API_URL,
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            print("ส่งข้อความ LINE สำเร็จ")

        else:
            print("ส่งข้อความ LINE ไม่สำเร็จ")
            print("Status Code:", response.status_code)
            print("รายละเอียด:", response.text)

    except requests.exceptions.RequestException as e:

        print("เกิดข้อผิดพลาดในการเชื่อมต่อ LINE")
        print(e)
        


# ==============================
# Test LINE Notification
# ==============================

if __name__ == "__main__":

    message = """
    AI SURVEILLANCE SYSTEM

    ระบบแจ้งเตือนพฤติกรรมต้องสงสัย

    สถานะ: ทดสอบระบบ
    ผลการทดสอบ: เชื่อมต่อ LINE สำเร็จ

    กรุณาตรวจสอบระบบ
    """

    send_line_message(message)