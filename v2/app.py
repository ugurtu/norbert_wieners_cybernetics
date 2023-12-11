from flask import Flask, render_template, request, jsonify
import chatbot
import gunicorn
import re
app = Flask(__name__)
# pip list --format=freeze > requirements.txt
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    # Simple regex to allow only alphanumeric characters and some punctuation
    if re.match("^[a-zA-Z0-9 .,!?\-:]+$", msg) and (len(msg) < 128):
        input = re.escape(msg)  # Escaping special characters
        input = input.replace("\\","")
        return get_Chat_response(input)
    else:
        return "Invalid input"


def get_Chat_response(text):
    # Let's chat for 5 lines
    return chatbot.retrieve_information(text)

"""
if __name__ == '__main__':
    app.run()
"""