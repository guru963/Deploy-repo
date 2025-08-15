# from flask import Flask
# from blueprints.salaried_bp import salaried_bp
# from blueprints.student_bp import student_bp
# from blueprints.farmer_bp import farmer_bp

# app = Flask(__name__)

# # Register Blueprints with URL prefixes to avoid route clashes
# app.register_blueprint(salaried_bp, url_prefix='/salaried')
# app.register_blueprint(student_bp, url_prefix='/student')
# app.register_blueprint(farmer_bp, url_prefix='/farmer')

# if __name__ == '__main__':
#     print("🚀 Running unified Flask app with Salaried, Student, and Farmer modules...")
#     app.run(debug=True, port=5000)
