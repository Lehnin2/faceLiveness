from flask import Blueprint
bp = Blueprint('liveness', __name__)
from app.liveness import routes