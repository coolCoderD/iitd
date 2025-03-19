from flask import Blueprint, request
from  controllers.report_controller import update_kb, embed_report, generalReportQuery, dateValQuery

report_bp = Blueprint('report_bp', __name__)

@report_bp.route('/update_kb', methods=['POST'])
def reportsUpdate():    
    return update_kb()

@report_bp.route('/embed_report', methods=['POST'])
def reportsEmbed():
    return embed_report()

@report_bp.route('/generalReportQuery', methods=['POST'])
def reportQuery():    
    return generalReportQuery(request)

@report_bp.route('/dateValQuery', methods=['POST'])
def date_val_Query():    
    return dateValQuery()