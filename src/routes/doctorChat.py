from flask import Blueprint # 
from  controllers.doctorChatController import chatController

doctor_chat_bp = Blueprint('doctor_chat_bp', __name__)

@doctor_chat_bp.route('/chat', methods=['POST'])
def chat():    
    return chatController()