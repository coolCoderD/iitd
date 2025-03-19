from flask import Blueprint
from  controllers.patientChatController import chatController

patient_chat_bp = Blueprint('patient_chat_bp', __name__)

@patient_chat_bp.route('/chat', methods=['POST'])
def chat():    
    return chatController()